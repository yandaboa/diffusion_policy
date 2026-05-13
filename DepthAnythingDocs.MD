# Depth Anything 3 (DA3) — Coding Agent Reference

A practical reference for writing code against ByteDance Seed's Depth Anything 3, a foundation model for visual geometry (monocular depth, multi-view depth, camera pose, 3D Gaussians) from one or more RGB images.

- **Repo:** https://github.com/ByteDance-Seed/Depth-Anything-3
- **Paper:** https://arxiv.org/abs/2511.10647
- **Hub org:** https://huggingface.co/depth-anything
- **License:** Apache 2.0 for the codebase; **per-model licenses vary** (see model table).

---

## 1. What DA3 Does

A single model predicts spatially consistent geometry from an arbitrary number of input images, with or without known camera poses. Capabilities depend on the chosen checkpoint:

- **Monocular depth** (relative or metric)
- **Multi-view depth** (fused, cross-view consistent)
- **Pose-conditioned depth** (when extrinsics/intrinsics are given as input)
- **Camera pose estimation** (extrinsics + intrinsics from images)
- **3D Gaussian estimation** (for novel view synthesis)

Backbone is a plain DINOv2 ViT; head is a depth-ray DPT variant. No architectural specialization per task — the input configuration determines behavior.

---

## 2. Model Selection

### Decision tree

```
Need metric depth (meters)?
├── Single image, monocular only      → DA3METRIC-LARGE
├── Multi-view or pose estimation too → DA3NESTED-GIANT-LARGE-1.1
└── No → Need pose / multi-view geometry?
        ├── Yes, best quality          → DA3-GIANT-1.1
        ├── Yes, balanced              → DA3-LARGE-1.1
        ├── Yes, lightweight           → DA3-BASE or DA3-SMALL
        └── No, just relative mono     → DA3MONO-LARGE
```

### Model table

| HF Repo ID | Params | Rel. Depth | Pose Est. | Pose Cond. | 3DGS | Metric Depth | Sky Seg | License |
|---|---|---|---|---|---|---|---|---|
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| `depth-anything/DA3-GIANT-1.1` | 1.15B | ✅ | ✅ | ✅ | ✅ | — | — | CC BY-NC 4.0 |
| `depth-anything/DA3-LARGE-1.1` | 0.35B | ✅ | ✅ | ✅ | — | — | — | CC BY-NC 4.0 |
| `depth-anything/DA3-BASE` | 0.12B | ✅ | ✅ | ✅ | — | — | — | Apache 2.0 |
| `depth-anything/DA3-SMALL` | 0.08B | ✅ | ✅ | ✅ | — | — | — | Apache 2.0 |
| `depth-anything/DA3METRIC-LARGE` | 0.35B | ✅ | — | — | — | ✅ | ✅ | Apache 2.0 |
| `depth-anything/DA3MONO-LARGE` | 0.35B | ✅ | — | — | — | — | ✅ | Apache 2.0 |

### Critical notes

- **Always prefer `-1.1` checkpoints** over the originals. Pre-1.1 weights were trained with a bug, especially affecting street scenes. The original `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, `DA3-LARGE` remain available but are **deprecated**.
- **CC BY-NC 4.0 = non-commercial only.** For commercial use, restricted to: `DA3-BASE`, `DA3-SMALL`, `DA3METRIC-LARGE`, `DA3MONO-LARGE`.

---

## 3. Installation

Requirements: Python ≥ 3.10, CUDA-capable GPU, PyTorch ≥ 2.

```bash
pip install xformers torch>=2 torchvision
pip install -e .                                              # basic
pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70  # for 3DGS head
pip install -e ".[app]"                                       # + Gradio UI
pip install -e ".[all]"                                       # everything
```

**Older GPUs without xformers support:** see Issue #11 in the repo for a patch.

---

## 4. Python API

### Basic inference

```python
import glob, torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.to(device=device)

images = sorted(glob.glob("path/to/images/*.png"))  # list[str] of paths, OR
# images can also be a list of np.ndarray (H, W, 3) uint8, or PIL.Image

prediction = model.inference(images)
```

### Prediction object — output shapes

| Attribute | Shape | Dtype | Notes |
|---|---|---|---|
| `prediction.processed_images` | `[N, H, W, 3]` | `uint8` | Images after model's preprocessing/resize |
| `prediction.depth` | `[N, H, W]` | `float32` | Per-pixel depth; **units depend on model** (see §5) |
| `prediction.conf` | `[N, H, W]` | `float32` | Per-pixel confidence |
| `prediction.extrinsics` | `[N, 3, 4]` | `float32` | World-to-camera, OpenCV/COLMAP convention |
| `prediction.intrinsics` | `[N, 3, 3]` | `float32` | Standard pinhole K matrix |

`N` = number of input images. `H, W` are the processed (not original) image dimensions.

### Useful inference flags

```python
prediction = model.inference(
    images,
    use_ray_pose=True,         # derive pose from ray head; slightly slower, more accurate
    # ref_view_strategy=...    # multi-view reference selection; see docs/funcs/ref_view_strategy.md
)
```

**`use_ray_pose` accuracy gain (AUC3, DA3NESTED-GIANT-LARGE):**

| Dataset | ray_head | cam_head |
|---|---|---|
| HiRoom | 84.4 | 80.3 |
| ETH3D | 52.6 | 48.4 |
| 7Scenes | 29.5 | 28.5 |
| ScanNet++ | 89.4 | 85.0 |

Default is `False` for speed.

---

## 5. Depth Units — Critical

| Model | What `prediction.depth` contains | Convert to meters |
|---|---|---|
| `DA3NESTED-GIANT-LARGE(-1.1)` | **Meters already** | No conversion needed |
| `DA3METRIC-LARGE` | Focal-normalized raw output | `metric_m = focal_px * depth / 300.0` |
| `DA3MONO-LARGE`, `DA3-GIANT/LARGE/BASE/SMALL` | **Relative** (affine-invariant) | Requires external scale+shift alignment |

### Metric conversion for `DA3METRIC-LARGE`

```python
fx, fy = K[0, 0], K[1, 1]            # from your camera intrinsics
focal_px = (fx + fy) / 2.0
metric_depth_m = focal_px * prediction.depth / 300.0
```

**Gotcha:** if you resize the input image before passing to DA3, the effective focal length scales linearly. Resizing width 1280 → 640 halves `fx`. Adjust accordingly, or pass the resized image's intrinsics.

### Aligning relative depth (Mono / non-metric models) to a metric reference

Solve for affine parameters `a, b` such that `a * d_rel + b ≈ d_metric` on valid reference pixels (e.g., from a depth sensor):

```python
import numpy as np
valid = (d_ref > 0) & np.isfinite(d_rel)
A = np.stack([d_rel[valid], np.ones(valid.sum())], axis=1)
a, b = np.linalg.lstsq(A, d_ref[valid], rcond=None)[0]
d_aligned = a * d_rel + b
```

Use **median** or RANSAC for the scale ratio if sensor data has outliers (e.g., flying pixels at depth discontinuities).

---

## 6. CLI

Set up once, reuse the loaded model via a backend process:

```bash
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE-1.1
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# Start backend (keeps model resident on GPU)
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR}

# Process a folder of images
da3 auto path/to/images \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/scene1 \
    --use-backend

# Process a video
da3 video path/to/video.mp4 \
    --fps 15 \
    --export-format glb-feat_vis \
    --export-dir ${GALLERY_DIR}/video1 \
    --process-res-method lower_bound_resize \
    --use-backend
```

**Export formats:** `glb`, `npz`, depth images, `ply`, 3DGS videos, feature visualizations.

Full CLI docs: `docs/CLI.md` in the repo.

---

## 7. Long Video — DA3-Streaming

For inference on long video sequences with bounded memory, use the streaming variant. Sliding-window inference keeps GPU memory **under 12 GB** even for ultra-long sequences. See `da3_streaming/README.md` in the repo.

---

## 8. VRAM Guidance

Approximate inference VRAM at default resolution (one image, fp16/bf16, xformers enabled):

| Model | Single image | 8 views | 16+ views |
|---|---|---|---|
| DA3-SMALL / BASE | 2–3 GB | 3–5 GB | 5–8 GB |
| DA3METRIC-LARGE / MONO-LARGE / LARGE-1.1 | 3–5 GB | 6–10 GB | 10–16 GB |
| DA3-GIANT-1.1 / NESTED-GIANT-LARGE-1.1 | 8–15 GB | 15–24 GB | use streaming |

Multi-view scales roughly with N² (global cross-view attention). For >16 views on consumer GPUs, prefer DA3-Streaming or reduce resolution via `--process-res-method`.

**Memory-saving knobs:**
- bf16/fp16 autocast (default in `model.inference`)
- Lower processing resolution
- xformers attention (mandatory for the giant models on 24 GB)
- For batch sequences, DA3-Streaming with sliding window

---

## 9. Custom Configs

Model architecture is YAML-driven. Configs live in `src/depth_anything_3/configs/`.

```python
from depth_anything_3.cfg import create_object, load_config
model = create_object(load_config("path/to/your/config.yaml"))
```

Example config skeleton:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params
  name: vitb
  out_layers: [5, 7, 9, 11]
  alt_start: 4
  qknorm_start: 4
  rope_start: 4
  cat_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params
  dim_in: 1536
  output_dim: 2
  features: 128
  out_channels: [96, 192, 384, 768]
```

---

## 10. Common Pitfalls

1. **Wrong checkpoint family.** Using the original (pre-1.1) Giant/Large weights silently gives worse results — especially outdoors. Always use `-1.1`.
2. **Forgetting metric conversion.** `DA3METRIC-LARGE` output is not in meters until you apply `focal * net_output / 300.`.
3. **Focal length after resize.** Adjust `fx, fy` by the resize factor, or pass intrinsics matching the resized image.
4. **Multi-view OOM.** Cross-view attention is global. Limit N or use streaming.
5. **Pose convention.** `extrinsics` is world-to-camera (OpenCV/COLMAP), not camera-to-world. Invert if you need c2w.
6. **`use_ray_pose=False` by default.** For best pose accuracy, set `True`.
7. **Commercial use of CC BY-NC models.** Giant/Large/Nested-Giant cannot be used commercially. Use Base/Small/Metric-Large/Mono-Large instead.
8. **xformers missing on older GPUs.** Without it, attention memory blows up. Apply the Issue #11 workaround.

---

## 11. Useful Repo Paths

- `src/depth_anything_3/api.py` — `DepthAnything3` user-facing class
- `src/depth_anything_3/model/da3.py` — `DepthAnything3Net` architecture
- `src/depth_anything_3/configs/` — YAML model configs
- `docs/CLI.md` — full CLI reference
- `docs/API.md` — full Python API reference
- `docs/BENCHMARK.md` — evaluation pipeline (5 datasets)
- `docs/funcs/ref_view_strategy.md` — multi-view reference selection
- `da3_streaming/` — long-video streaming variant
- `notebooks/` — example notebooks

---

## 12. Citation

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen
          and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
