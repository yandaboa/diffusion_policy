# STATE policy — real-robot pipeline

Everything needed to calibrate, visualize, and evaluate the **low-dim state policy** (RSL-RL
`rl_state_cfg.py` `PolicyCfg`, TorchScript-exported) on the real UR5e + Robotiq 2F-85.

The policy never sees an image. It consumes a 215-dim vector built from robot proprioception
plus the **peg** and **peg-hole** poses, which are recovered from AprilTags by a separate
perception process. Almost all of the work below is making those two object poses correct in
the robot base frame.

> **⚠️ The repo is currently broken for this pipeline.** Commit `9a8868d` ("resolved merge
> conflict") committed *unresolved* conflict markers. `eval_real_robot.py`,
> `diffusion_policy/real_world/real_env.py`, `scripts/sim2real/0_camera_calibrate.py`,
> `rtde_interpolation_controller.py`, and `pointnet_policy.py` do not parse.
> See [Known breakage](#known-breakage) before running anything.

---

## 1. Architecture

Three processes, loosely coupled through files. The eval never owns a camera.

```
 ┌─ peg_fusion_viz.py --publish --headless ──────────────┐   foundstereo env
 │   owns front D455 + side D435 + wrist D415            │
 │   AprilTag PnP per camera → lift to base → robust fuse│
 │   also accumulates the STATIC peg-hole pose           │
 └──────────────────┬────────────────────────────────────┘
                    │ atomic JSON, ~30 Hz
                    ▼
        Log/peg_twin_state.json   {stamp, capture_stamp, joints,
                                   T_base_peg, ncams, cams{}, T_base_hole, hole{}}
                    │                                  │
   ┌────────────────┴──────────────┐    ┌──────────────┴───────────────────────┐
   ▼                               │    ▼                                       │
 eval_real_robot.py                │  live_digital_twin.py  (Isaac, UWLab repo) │
   camera-less RealEnv             │    renders the sim twin from the same state │
   PublishedPegPose reader         │            │ /dev/shm/twin_frame.npy        │
   215-dim obs → JIT policy → OSC  │            ▼                                │
                                   │  twin_frame_viewer.py  (combined GUI) ◄─────┘
                                   └─ /dev/shm/real_front_frame.npy
```

Why decoupled: pose estimation must never block the 10 Hz control loop, and the fusion must be
*literally* the tested `peg_fusion_viz` code (calibrated resolutions, per-serial intrinsics with
distortion, prior-hold, robust merge) rather than a re-implementation that can silently diverge.

**Environments** — `peg_fusion_viz.py`, `calibrate_hole_tags.py`, `pc_overlay_align.py`,
`twin_frame_viewer.py` all run in **`foundstereo`** (it owns `pyorbbecsdk`). `eval_real_robot.py`
runs in `foundstereo` or `robodiff_real` (both have torch + rtde + imageio). The sim-side scripts
under `~/UWLab-patrick-private/scripts_v2/tools/sim2real/` run in the **Isaac** env (`env_uwlab`),
with `CUDA_VISIBLE_DEVICES=1` (GPU0 is usually full).

---

## 2. Hardware inventory

| Role | Model | Serial | Mount | Extrinsic source |
|---|---|---|---|---|
| front | RealSense D455 | `215122255213` | stationary | `calibrations/most_recent_hand_aligned_extrinsic_215122255213.json` |
| side | RealSense D435 | `832112070487` | stationary | `..._832112070487.json` |
| wrist | RealSense D415 | `746112060198` | eye-in-hand | `wrist_cam_in_ee.npy` + live FK |
| *(retired)* | Orbbec Femto Bolt | `CL838420160` | was front | `..._CL838420160.json` |

Color resolutions the extrinsics were calibrated at: **front 1280×720**, **side/wrist 1920×1080**.
`peg_fusion_viz` opens them at exactly these by default — changing `--front-res` / `--side-res` /
`--wrist-res` invalidates the intrinsics scaling assumptions.

**Tags.** Peg = AprilTag `16h5`, IDs 0–5, **25 mm**, one centred per cube face
(`apriltag_peg_pose.FACES`, peg bbox 30×30×60 mm). Peg-hole = `16h5`, IDs **7, 8, 9**, **26 mm**,
glued at arbitrary measured poses (`tag_body.TagBody`), block bbox 69.85×69.85×40.64 mm.

---

## 3. Calibration

Do these in order. Steps 1–3 are per-camera and only repeat when a camera physically moves;
step 4 repeats when the hole fixture or its tags move.

### 3.1 Camera → ChArUco board

```bash
conda activate foundstereo
python scripts/sim2real/0_camera_calibrate.py --camera realsense --serial 215122255213 --board charuco
```

Place the 11×11 ChArUco board (square 55.4 mm, marker 40 mm, `DICT_4X4_100`) with its **top-left
corner under the centre of the UR5 base plate** — see `scripts/sim2real/ALIGN_SCENE.MD`. Writes a
timestamped `<date>_charuco.json` **and** `calibrations/most_recent_charuco_calib.json`.

`--serial` opens only that one camera; without it, `gather_realsense_cameras` opens depth+color on
*every* RealSense and three depth streams stall the USB bus.

### 3.2 Board → robot base (hand alignment)

The board corner frame is not exactly the sim base frame. Nudge a rigid correction until the live
cloud lands on a sim reference cloud; `T_total = T_corr @ T_charuco` **is** camera→base.

```bash
# (a) capture the sim reference cloud, Isaac env, in ~/UWLab-patrick-private:
CUDA_VISIBLE_DEVICES=1 python scripts_v2/tools/sim2real/scene_capture.py \
    --enable_cameras --camera orbbec --idx 3 --save_pc

# (b) drive the real arm to that same reset config:
python sim_overlay_align.py <sim.png> --robot_ip 192.168.1.10 \
    --joints 0.363852,-1.338813,2.095192,-2.831392,-1.785279,0.331381

# (c) align, foundstereo env:
python pc_overlay_align.py \
    --sim_pc /home/yandabao/UWLab-patrick-private/scene_capture_front_orbbec_idx3_pc_base.npy \
    --calib scripts/sim2real/perception/calibrations/most_recent_charuco_calib.json \
    --camera realsense --serial 832112070487
```

Keys: `dawsec` translate, `ljikou` rotate, `789` flip 180° about X/Y/Z, `=-`/`][` step size,
`f` freeze, `x` crop, `r` reset, **`p` save**, ESC quit.

`p` writes **two** files: the permanent per-serial
`most_recent_hand_aligned_extrinsic_<serial>.json` *and* the unkeyed
`most_recent_hand_aligned_extrinsic.json`, which is **overwritten by whichever camera you aligned
last**. Fusion and eval read the *per-serial* files — never the unkeyed one.

The clean (non-`--occlude`) `_pc_base.npy` is full robot geometry in the base frame and is
**camera-independent**, so the front capture works for the side camera too. The arm must be at the
same joints as the capture. Resume an existing alignment with `--resume <extrinsic.json>` instead
of `--calib`.

### 3.3 Wrist camera → wrist_3_link

The wrist camera is eye-in-hand: `T_base_cam(t) = FK(q_t) @ T_wrist3link_cam`. Extract the fixed
part from sim **physics** (not the rendered `cam.data.pos_w`, which lags physics):

```bash
cd ~/UWLab-patrick-private
CUDA_VISIBLE_DEVICES=1 python scripts_v2/tools/sim2real/get_wrist_cam_in_ee.py
```

Produces `wrist_cam_in_ee.npy` (4×4 f64, OpenCV convention; camera origin in wrist_3_link ≈
`[-0.003, -0.067, +0.034]` m). `peg_fusion_viz` looks for it in `perception/calibrations/` first,
then falls back to `~/UWLab-patrick-private/wrist_cam_in_ee.npy` (where it currently lives).

### 3.4 Peg-hole tags

The policy consumes `in_receptive = inv(T_base_hole) @ T_base_peg`. A residual extrinsic error
that left-multiplies *both* cancels — **but only if the hole was localized through the same
tag+extrinsic path as the peg**. So the hole is never measured with calipers; it is derived from
the peg tracker.

Seated, the peg is almost entirely inside the block and its side tags are occluded, so the grasp
carries the pose in via FK. Three prompted phases: (A) hold the gripped peg in free view across
several poses → `T_ee_peg`; (B) seat it, then
`T_base_hole = FK(q) @ T_ee_peg @ Trans(0,0,-14.837 mm)` and solve each `T_hole_tag`; (C) lift out
and re-measure the grasp — if the peg slipped during seating the run **aborts without saving**.

```bash
conda activate foundstereo            # peg_fusion_viz must be STOPPED — this owns the cameras
python scripts/sim2real/perception/calibrate_hole_tags.py --identify        # which IDs are on it?
python scripts/sim2real/perception/calibrate_hole_tags.py \
    --robot_ip 192.168.1.10 --ids 7,8,9 --hole-tag-size 0.026
```

Useful flags: `--frames 150`, `--verify-frames 40`, `--grasp-frames 60`, `--max-slip-mm 1.0`,
`--min-peg-tags 2`, `--min-tag-obs 20`, `--settle 3.0`, `--no-grasp-carry`, `--out <path>`.

Writes `calibrations/hole_tags.json` — `{name, aruco_dict, tag_size_m, dims_m, tags{id: T_body_tag},
calib{...}}`. The `calib.verify` block is the acceptance test; the sim success gate is 2.5 mm /
0.025 rad, so verification errors must land well inside that.

**Current calibration on disk** (2026-08-05): tags 7/8/9, verify `pos_err 0.39 mm`,
`rot_err 0.036°`, `jitter 0.27 mm / 0.088°`, grasp slip `0.16 mm`, `passed: true`. Per-tag spread
is uneven — id 9 (side cam) `1.2 mm`, id 7 (front cam) `12.7 mm`, id 8 (both) `30.6 mm`, which is
the front camera's known bias showing up again (§7).

---

## 4. The perception worker

```bash
conda activate foundstereo
python scripts/sim2real/perception/peg_fusion_viz.py --publish --headless --robot_ip 192.168.1.10
```

Each camera estimates `T_cam_peg` from the peg's tags, then it is lifted into the base frame —
front/side by their fixed extrinsic, wrist by `FK(getActualQ()) @ T_wrist3link_cam` — and the
estimates are fused. Without `--headless` you get an Open3D scene (front=red, side=green,
wrist=blue, fused centroid white) and the console prints per-camera base position plus the
agreement spread, **which is the real cross-camera accuracy check**.

Pose estimation (`apriltag_peg_pose.estimate_peg_pose_ex`):
- **≥2 tags** → one joint `solvePnP(SQPNP)+refineLM` over all corners lifted into the peg frame.
  Non-coplanar faces kill the single-tag flip. ~0.65° / 1.8 mm median.
- **1 tag** → `solvePnPGeneric(IPPE_SQUARE)`, branch picked nearest the previous frame's prior;
  confidence down-weighted hard (planar 2-fold ambiguity).
- Fusion is a confidence-weighted geodesic **median** (IRLS) at ≥3 estimates, a weighted **mean**
  at exactly 2 — the median is degenerate with two samples and made the fused peg teleport.

| Flag | Default | Notes |
|---|---|---|
| `--publish [PATH]` | `Log/peg_twin_state.json` | atomic write each loop |
| `--headless` | off | no Open3D window |
| `--robot_ip` | `192.168.1.10` | needed for wrist FK; auto-disables if unreachable |
| `--no-wrist` | off | skip wrist camera + FK |
| `--fuse-exclude` | `""` | drop a camera from the centroid (still drawn/published) |
| `--prior-hold N` | `15` | keep the flip-disambiguation prior alive through dropouts |
| `--allow-single-tag` | off | by default single-tag cams are excluded when any cam sees ≥2 |
| `--pose-mode` | `joint` | `per-tag` = old behavior |
| `--merge-mode` | `median` | `mean` = old equal-weight behavior |
| `--publish-hz` | `30` | |
| `--front/side/wrist-res` | calibrated | see §2 — don't change casually |
| `--hole-tags [PATH]` | auto | uses `hole_tags.json` when it exists; `--no-hole-tags` disables |
| `--hole-min-tags` | `2` | a single planar hole tag flips |
| `--hole-scan-interval` | `0.5 s` | the hole is static, so scanning is deliberately cheap |
| `--hole-window` | `120` | observations retained per camera for the robust average |
| `--record-control PATH` | none | poll a control file and save camera videos (the eval sets this) |
| `--front-frame [PATH]` | `/dev/shm/real_front_frame.npy` | feeds the combined twin viewer |
| `--refine-front` | off | **see the warning in §7 before using** |

### Published state file

```jsonc
{
  "stamp":         1785974862.2,   // publish time
  "capture_stamp": 1785974862.1,   // OLDEST camera frame that produced this pose → true latency
  "joints":        [q0..q5],
  "T_base_peg":    [[..4x4..]],    // fused centroid
  "ncams":         2,
  "cams":  { "front": {"T": .., "seen": .., "tags": [ids], "capture_stamp": .., "detect_ms": .., "used_roi": ..}, ... },
  "T_base_hole":   [[..4x4..]],    // accumulated static hole pose (omitted if hole tracking off)
  "hole":          {"n": .., "spread_mm": .., "spread_deg": .., "cams": [..], "tags": [ids]}
}
```

`diffusion_policy/real_world/peg_pose_reader.py::PublishedPegPose` is the latest-value reader.
A peg read counts as `seen` only if `stamp` is within `stale_after_s` **and** `T_base_peg` is
present; otherwise the consumer holds its last pose. `T_base_hole` is deliberately **not**
staleness-gated — the hole cannot move, so the latest accumulated value is always the best one.

---

## 5. Running the eval

```bash
# terminal 1 — perception (foundstereo)
python scripts/sim2real/perception/peg_fusion_viz.py --publish --headless --robot_ip 192.168.1.10

# terminal 2 — eval
python eval_real_robot.py \
    -i /home/yandabao/UWLab-patrick-private/pulled_ckpts/exported/policy.pt \
    -o debug_state/ --robot_ip 192.168.1.10
```

Or let the eval spawn the worker itself with `--launch_fusion` (it waits 6 s for the first publish
and terminates the worker at exit); pass extra worker args through `--fusion_extra "--fuse-exclude front"`.

**Controls** (click the OpenCV window first): `C` start · `S` stop · `R` reset + relabel + restart ·
`G` open gripper macro · `Q` quit. Episodes auto-terminate on EE `z > --z_terminate` or
`--max_duration`, then prompt `s`=success / `f`=fail.

| Option | Default | Notes |
|---|---|---|
| `-i / --input` | — | TorchScript state policy `.pt` |
| `-o / --output` | — | results dir |
| `-ri / --robot_ip` | — | |
| `-j / --init_joints` | off | move to the initial joint config at startup |
| `-f / --frequency` | `10` Hz | |
| `-md / --max_duration` | `90` s | |
| `--z_terminate` | `0.4` m | EE z in base frame |
| `--peg_state_file` | `Log/peg_twin_state.json` | |
| `--peg_stale_s` | `0.5` | older ⇒ treated as unseen, last pose held |
| `--launch_fusion` / `--fusion_extra` | off / `""` | auto-spawn the worker |
| `--hole_from_tags` / `--no_hole_from_tags` | **on** | latch the hole pose per episode |
| `--hole_min_samples` | `8` | below this, fall back to the configured pose |
| `--hole_pose` / `--hole_pose_file` | none | fallback fixed hole pose |
| `--save_video` / `--no_save_video` | **on** | status canvas + worker-side camera videos |
| `--debug_obs` | off | ~1 Hz dump of the named obs blocks |
| `--action_noise` | `0.0` | Gaussian std on raw arm actions, pre-scale |
| `--torch_device` | `cuda` | |

**Hole pose resolution.** With `--hole_from_tags` (default) the worker's accumulated tag estimate
is latched **once at the start of every episode** and then frozen — the hole is bolted down, so
re-reading it each step would only inject tag jitter into `receptive` / `in_receptive`, and a bump
between episodes is picked up by the next latch. It falls back to `--hole_pose_file`,
`--hole_pose "x,y,z,rx,ry,rz"`, or the hardcoded `DEFAULT_HOLE_POSE =
[0.5400, 0.225, -0.005, 1, 0, 0, 0]` when fewer than `--hole_min_samples` observations exist.

**Output layout**

```
<output>/
  eval_results.json               {n_episodes, n_success, success_rate, episodes[]}
  policy_status_ep_000.mp4        the status canvas
  videos/record_control.json      control file the worker polls
  videos/<episode>/<cam_idx>.mp4  camera videos, written by the worker
```

### Observation vector (215 dims)

Isaac `ObservationManager` with `concatenate_terms=True`, `flatten_history_dim=True`,
`history_length=5`: each term is flattened over its 5-frame history **oldest→newest**, then the
blocks are concatenated in cfg term order. Short histories back-fill by repeating the earliest
frame (`CircularBuffer` behavior).

| # | Term | Dim | Content |
|---|---|---|---|
| 1 | `insertive_asset_in_receptive_asset_frame` | 6 | peg in peg-hole frame |
| 2 | `prev_actions` | 7 | `[dx,dy,dz,drx,dry,drz,grip]` — **raw** policy output, pre-scale |
| 3 | `joint_pos` | 12 | 6 arm + 6 Robotiq mimic (absolute) |
| 4 | `end_effector_pose` | 6 | wrist_3 in base |
| 5 | `insertive_asset_pose` | 6 | peg in wrist_3 frame |
| 6 | `receptive_asset_pose` | 6 | peg-hole in wrist_3 frame |

`(6+7+12+6+6+6) × 5 = 215`.

> The **annotated** `insertive_asset_in_receptive_asset_frame` configclass field is collected
> *before* the unannotated fields — that is why `in_receptive` leads rather than `prev_actions`.
> If you reorder `PolicyCfg`, `STATE_TERM_ORDER` in `eval_real_robot.py` must follow.

All pose terms are `pos(3) + axis-angle(3)`, using `subtract_frame_transforms` semantics
(`inv(T_base_root) @ T_base_target`) with the rotation as `R.from_matrix(...).as_rotvec()`.
`compute_calibrated_ee_pose` uses scipy's canonical shortest-angle rotvec to match Isaac's
`axis_angle_from_quat`; the older `quat_to_axis_angle` does not canonicalize sign and can differ
by 2π when `q.w < 0`.

Gripper mimic: `master_angle = gripper_pos_raw × π/4`, then
`GRIPPER_MIMIC_RATIOS = [+1,+1,-1,+1,-1,-1] × master_angle`.

`prev_action` comes from `env.get_last_action()`. `RealEnv` owns `last_action` and updates it on
**every** `exec_actions` call — unconditionally, even when scheduling filtered the execution — so
the term never stalls; it is zeroed on `start_episode`. The eval schedules each action strictly in
the future, because `exec_actions` drops actions with past timestamps.

The **gripper-open macro** and stuck detection (no joint moves >0.002 rad for 2 s → force open)
affect the *executed* gripper only. The saved `prev_action` keeps the policy's own raw output.

### Actions

Eval `RelCartesianOSC` (`Ur5eRobotiq2f85RelativeOSCEvalAction`), 6 arm dims + binary gripper:

```
CARTESIAN_SCALE = [0.01, 0.01, 0.002, 0.02, 0.02, 0.2]     # m, m, m, rad, rad, rad
raw_arm × scale → apply_delta_pose(get_ee_pose(q)) → absolute target
gripper: raw < 0 closes
```

The checkpoint may carry a sidecar `<stem>_meta.txt`; if it declares `num_proprio` and it isn't
215, the eval refuses to start.

---

## 6. Digital twin

Three processes sharing the same `Log/peg_twin_state.json` the eval reads, so the twin can run
**live during an eval**.

```bash
# 1. peg-pose worker — foundstereo (owns the cameras)
python scripts/sim2real/perception/peg_fusion_viz.py --publish --headless --robot_ip 192.168.1.10 \
    --front-frame                      # also feed the real-front-cam tile

# 2. sim twin renderer — Isaac env, from ~/UWLab-patrick-private
CUDA_VISIBLE_DEVICES=1 python scripts_v2/tools/sim2real/live_digital_twin.py \
    --stream_frames /dev/shm/twin_frame.npy
    # --state_file defaults to /home/yandabao/diffusion_policy/Log/peg_twin_state.json

# 3. combined viewer — foundstereo, needs a display
python scripts/sim2real/perception/twin_frame_viewer.py --scale 0.9
```

The viewer composes the sim overview plus front/side/wrist tiles with per-camera peg overlays,
showing the real front camera large beside the sim front camera. Flags: `--frame`, `--real-front`,
`--state-file`, `--scale`, `--hz 30`, `--peg-stale 1.0`.

Plain IsaacLab twin window, no combined GUI:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts_v2/tools/sim2real/live_digital_twin.py --enable_cameras
CUDA_VISIBLE_DEVICES=1 python scripts_v2/tools/sim2real/live_digital_twin.py --enable_cameras --livestream 2   # → browser
```

Other visualization / diagnosis tools:

- `scripts/sim2real/perception/debug_fusion_jitter.py` — hold the peg still, measure per-camera
  source jitter (joint vs per-tag) and fused jitter. This is what isolated the front camera.
- `scripts/sim2real/perception/peg_fusion_viz.py --tag-view` — per-camera tag detection overlays.
- `sim_overlay_align.py` — blend a sim render over the live feed to match the *scene*.
- `pc_overlay_align.py` — the point-cloud analogue, and the actual extrinsic aligner.

---

## 7. Known issues and gotchas

**Front camera (D455) has a real systematic extrinsic bias.** Restricted to frames where the front
sees ≥2 tags, front-vs-side base offset is `[+20, −6, +2.5] mm` (‖21 mm‖) plus 4.8°, with std
<1 mm / 0.2° — rigid, not noise. The front also sees only one tag ~76 % of the time, where IPPE's
planar ambiguity gives 20–35° flips. **Recommended until it is physically re-fixed:**
`--fuse-exclude front`. Side alone measures 0.06° / 0.13 mm.

**`--refine-front` has burned us.** Collection was contaminated by single-tag flips (the static
gate passes when both bracketing reads flip the same way) and the save guard only checked
position, so four bad corrections were stacked. It is now hardened — front must see ≥2 tags in
both bracketing reads, saves are rejected if position *or* rotation worsens, and implausible
corrections (>60 mm or >15°) are refused. Recovery, if it happens again: the clean pre-refine
backup is the only `..._prerefine_*.json` with **no** `refined` key (front pos ≈ `[1.024, −0.175,
0.425]`). Prefer redoing the front `pc_overlay_align`.

**D435 stuck startup.** After an unclean Ctrl-C the D435 often hangs in `wait_for_frames` forever.
There is a warmup that drops initial frames; if it delivers zero, unplug/replug. Also never open
depth on all three RealSenses at once — use `--serial`.

**Peg-pose staleness.** The eval combines a possibly-milliseconds-stale published `T_base_peg`
with the *current* EE pose, which introduces a small grasp-transform error during fast motion.
Acceptable given the worker's rate. The state file also carries `joints`, which would allow a
staleness-robust grasp transform — not implemented.

**The unkeyed extrinsic file is a trap.** `most_recent_hand_aligned_extrinsic.json` is overwritten
by the last camera aligned. Always read the per-serial file.

**Peg unseen.** If no peg pose has ever arrived, `insertive` and `in_receptive` are **zeros** (and
the eval warns once). Verify `[fusion] peg pose live` at startup; use `--debug_obs` to confirm
`insertive` is small and stable (it should be, since the peg is grasped).

### Known breakage

Commit `9a8868d` committed unresolved conflict markers into tracked files. These do not parse:

| File | Conflicts |
|---|---|
| `eval_real_robot.py` | 6 — CLI options, `dt`, status canvas, reset/terminate handlers |
| `diffusion_policy/real_world/real_env.py` | 9 — incl. a ~190-line block at 956–1147 |
| `scripts/sim2real/0_camera_calibrate.py` | 2 |
| `diffusion_policy/real_world/rtde_interpolation_controller.py` | 1 |
| `diffusion_policy/real_world/pointnet_policy.py` | 1 |
| `scripts/sim2real/ALIGN_SCENE.MD` | 1 |

In every conflict the **`HEAD` side is the state-policy version** documented here; the
`e45600ad` side is the older RGB/diffusion-checkpoint eval (`match_dataset`, `--cameras`,
`get_real_obs_resolution`, hydra/dill checkpoint loading). Resolving toward `HEAD` and deleting
the `e45600ad` blocks restores this pipeline. Check with:

```bash
grep -rn "^<<<<<<< \|^>>>>>>> " --include="*.py" --include="*.md" --include="*.MD" . \
  | grep -v -e Depth-Anything -e FoundationPose -e segment-anything -e Fast-Found
python -c "import ast; ast.parse(open('eval_real_robot.py').read())"
```

---

## 8. File map

| Path | Role |
|---|---|
| `eval_real_robot.py` | the state-policy eval (this doc) |
| `diffusion_policy/real_world/peg_pose_reader.py` | `PublishedPegPose` latest-value reader |
| `diffusion_policy/real_world/real_env.py` | `get_obs(modality='rgb'\|'state'\|'pc'\|'pose')`; `get_obs_state` is camera-less |
| `diffusion_policy/real_world/ur5e_kinematics.py` | `get_ee_pose`, `forward_kinematics_calibrated`, `apply_delta_pose` |
| `scripts/sim2real/perception/peg_fusion_viz.py` | the fusion worker / publisher |
| `scripts/sim2real/perception/apriltag_peg_pose.py` | per-camera tag PnP (`estimate_peg_pose_ex`) |
| `scripts/sim2real/perception/async_peg_pipeline.py` | per-camera capture + detect threads, ROI tracking |
| `scripts/sim2real/perception/tag_body.py` | `TagBody`; peg-hole constants (`ASSEMBLED_OFFSET_Z`) |
| `scripts/sim2real/perception/calibrate_hole_tags.py` | peg-anchored hole-tag calibration |
| `scripts/sim2real/perception/twin_frame_viewer.py` | combined twin GUI |
| `scripts/sim2real/perception/debug_fusion_jitter.py` | per-camera jitter diagnosis |
| `scripts/sim2real/0_camera_calibrate.py` | intrinsics + ChArUco extrinsic |
| `pc_overlay_align.py` | board→base hand alignment (produces the extrinsics) |
| `sim_overlay_align.py` | sim/real RGB overlay; also drives the arm to a joint pose |
| `scripts/sim2real/ALIGN_SCENE.MD` | scene-setup recipe for the above |
| `scripts/sim2real/DIGITAL_TWIN.md` | the three-process twin recipe |
| `~/UWLab-patrick-private/scripts_v2/tools/sim2real/` | sim side: `scene_capture.py`, `live_digital_twin.py`, `get_wrist_cam_in_ee.py` |

Sibling docs for the *other* evals: `POINTCLOUD_EVAL.md`, `EVAL_DEPTH.md`, `CAMERA_EXTRINSICS.md`,
`online_segmentation.md`. The history-conditioned tactile eval is `eval_real_robot_tactile.py`.
