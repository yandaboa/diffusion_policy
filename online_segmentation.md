# Online robot-arm segmentation during eval (design notes)

Status: **not implemented.** `segment_pointclouds.py` is the offline (post-process)
path used for calibration. This doc sketches how to do the same thing *live* during
eval, if/when the eval pipeline needs arm-only points online (e.g. as policy input or
for live perception-gap monitoring). Until then, keep recording raw clouds + joints
and segment offline.

## Why offline ≠ online here

The offline script uses `sam2_video_predictor`, which takes a **directory of all
frames** and does a forward+backward propagation pass — it needs the whole sequence
up front. Online means frames arrive one at a time and the mask must be produced
*before* the next frame, with no look-ahead.

The arm **moves** during eval, so we can't just reuse a single click each frame with
the plain image predictor — the mask would drift off the arm. We need SAM2's memory
bank in **streaming** mode: prompt once on the first frame, then `track()` each new
frame using accumulated memory.

The officially-installed `sam2` package in `foundstereo` has **no streaming/camera
predictor** (only `sam2_image_predictor` and the offline `sam2_video_predictor`). So
online requires one of the two options below.

## Hardware budget (already fine)

- 2× RTX 4090 (24 GB each), idle.
- FFS (8 iters, 848×480): ~30–60 ms/frame. SAM2 encoder: ~25–40 ms (base_plus), ~15 ms (tiny).
- FFS + SAM2 on one 4090 ≈ 60–100 ms → 10–16 FPS (clears the 10 Hz control loop).
- Better: FFS on GPU0, SAM2 on GPU1 → both ~30 FPS. Set per-process via `CUDA_VISIBLE_DEVICES`.

Speed is not the blocker; the missing streaming API is.

## Option A — real-time fork (recommended, least work)

Use the community streaming predictor (`SAM2CameraPredictor`), e.g.
`Gy920/segment-anything-2-real-time` or `facebookresearch/sam2` streaming branches.
It exposes a memory-based streaming API:

```python
from sam2.build_sam import build_sam2_camera_predictor
predictor = build_sam2_camera_predictor(model_cfg, ckpt, device='cuda:1')

# once, at eval start, on the first warmup frame:
predictor.load_first_frame(first_left_ir_rgb)
predictor.add_new_prompt(frame_idx=0, obj_id=1, points=clicks, labels=labels)

# then per frame in the hot loop:
obj_ids, mask_logits = predictor.track(left_ir_rgb)   # uses memory, no re-prompt
mask = (mask_logits[0, 0] > 0).cpu().numpy()
```

Install into `foundstereo` in a way that does **not** clobber the current `sam2`
(pin/venv or a separate import path), since the offline script depends on the
official package.

## Option B — adapt the official video predictor (no new dep)

`sam2_video_predictor` already has the streaming building blocks internally
(`init_state`, per-frame memory, `propagate_in_video` yields incrementally). Wrap it
to consume frames as they arrive instead of from a directory:

1. `init_state` on a tiny seed (first frame only).
2. `add_new_points_or_box` on frame 0 with the click(s).
3. For each new live frame, append it to the in-memory frame store and run a single
   propagation step using the existing memory (mirror the internal loop of
   `propagate_in_video`, one step at a time).

More work and depends on SAM2 internals, but avoids a second dependency.

## Integration into the eval pipeline

The two-process design already in place makes this clean:

- `depth_recorder.py` (in `foundstereo`) already grabs the D455 IR pair and runs FFS
  per frame. Add SAM2 there:
  - one-time: at warmup, write the first left-IR frame and **block for a click**
    (or accept `--point x,y` so eval can run unattended), feed it as the prompt.
  - per frame: `mask = predictor.track(left_rgb)` → `depth_arm = where(mask, depth, 0)`
    → back-project only arm pixels (same `depth2xyzmap` + z-filter as offline).
  - put SAM2 on `cuda:1` and FFS on `cuda:0` to keep both ~30 FPS.
- `action_playback_depth.py` (in `robodiff_real`) is unchanged — it still just
  coordinates via the ready/stop sentinels and shared wall clock.

## Caveats / gotchas

- **Prompt at eval start, not calibration time.** The scene/arm pose differs per run,
  so the first-frame click must happen live (or be supplied via `--point`). A fixed
  pixel won't generalize across setups.
- **Re-init on tracking loss.** If the arm leaves frame or the mask collapses, you
  need a re-prompt path; offline propagation hides this because it can look ahead.
- **VRAM coexistence.** FFS + SAM2 + (any policy net) on the same GPU — check headroom;
  splitting across the two 4090s sidesteps it.
- **Determinism.** Online (causal, memory-only) masks will differ slightly from the
  offline (bi-directional) masks. For calibration use offline; for live use online,
  and don't expect them to be pixel-identical.

## Recommendation

Only build this if the **policy consumes arm-only points online**. If segmentation is
purely for calibration / perception-gap analysis, record raw during eval and reuse
`segment_pointclouds.py` offline — simpler and lossless.
