"""
Run DA3METRIC-LARGE on side + wrist camera frames and save visualizations:
  - depth only (TURBO colormap)
  - depth + RGB overlay (50/50 blend)
  - side-by-side panel of both cameras

Output goes to depth_testing/da3_viz/.

Usage:
    conda run -n env_uwlab python3 save_da3_visualizations.py
"""

import sys, os, time
import numpy as np
import cv2
import torch

sys.path.insert(0, "/home/yandabao/diffusion_policy/Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3

SIDE_VIDEO  = "depth_testing/videos/0/1.mp4"
WRIST_VIDEO = "depth_testing/videos/0/2.mp4"
OUT_DIR     = "depth_testing/da3_viz"
MODEL_ID    = "depth-anything/DA3METRIC-LARGE"
DEVICE      = torch.device("cuda:0")

# D435 and D415 at 640×480 — approximate intrinsics (for metric conversion label only)
# fx/fy ~ 610 px for both cameras at this resolution
FOCAL_PX = {"side": 610.0, "wrist": 610.0}

N_FRAMES = 12   # number of evenly-spaced frames to save


def read_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)   # BGR, kept for overlay
    cap.release()
    return frames


def depth_to_turbo(depth_m: np.ndarray, d_min=0.0, d_max=3.0) -> np.ndarray:
    """float32 metric depth → BGR uint8 TURBO colourmap, clipped to [d_min, d_max]."""
    norm = np.clip((depth_m - d_min) / (d_max - d_min), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def annotate(img: np.ndarray, label: str, depth_m: np.ndarray) -> np.ndarray:
    valid = (depth_m > 0.01) & (depth_m < 3.0)
    mean_m = float(depth_m[valid].mean()) if valid.any() else float("nan")
    img = img.copy()
    cv2.putText(img, label,              (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, f"mean={mean_m:.2f}m", (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID} …")
    model = DepthAnything3.from_pretrained(MODEL_ID).to(device=DEVICE)
    model.eval()

    print(f"Reading {N_FRAMES} frames …")
    side_bgr  = read_frames(SIDE_VIDEO,  N_FRAMES)
    wrist_bgr = read_frames(WRIST_VIDEO, N_FRAMES)

    for i, (s_bgr, w_bgr) in enumerate(zip(side_bgr, wrist_bgr)):
        s_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)
        w_rgb = cv2.cvtColor(w_bgr, cv2.COLOR_BGR2RGB)

        pred = model.inference([s_rgb, w_rgb])

        # pred.depth shape: [2, H, W], focal-normalised → convert to metres
        # metric_m = focal_px * raw / 300.0
        s_depth_raw = np.asarray(pred.depth[0])
        w_depth_raw = np.asarray(pred.depth[1])

        s_depth_m = FOCAL_PX["side"]  * s_depth_raw / 300.0
        w_depth_m = FOCAL_PX["wrist"] * w_depth_raw / 300.0

        # Resize RGB to match model output resolution for overlay
        H, W = s_depth_raw.shape
        s_bgr_r = cv2.resize(s_bgr, (W, H))
        w_bgr_r = cv2.resize(w_bgr, (W, H))

        # ── Depth only ────────────────────────────────────────────────────
        s_depth_vis  = annotate(depth_to_turbo(s_depth_m), "SIDE  (DA3)", s_depth_m)
        w_depth_vis  = annotate(depth_to_turbo(w_depth_m), "WRIST (DA3)", w_depth_m)

        # ── Depth + RGB overlay ───────────────────────────────────────────
        s_overlay = cv2.addWeighted(depth_to_turbo(s_depth_m), 0.5, s_bgr_r, 0.5, 0)
        w_overlay = cv2.addWeighted(depth_to_turbo(w_depth_m), 0.5, w_bgr_r, 0.5, 0)
        s_overlay = annotate(s_overlay, "SIDE  (DA3 + RGB)", s_depth_m)
        w_overlay = annotate(w_overlay, "WRIST (DA3 + RGB)", w_depth_m)

        # ── Panels ────────────────────────────────────────────────────────
        depth_panel   = np.concatenate([s_depth_vis, w_depth_vis], axis=1)
        overlay_panel = np.concatenate([s_overlay,   w_overlay],   axis=1)
        full_panel    = np.concatenate([depth_panel, overlay_panel], axis=0)

        cv2.imwrite(f"{OUT_DIR}/frame{i:02d}_depth_only.jpg",   depth_panel)
        cv2.imwrite(f"{OUT_DIR}/frame{i:02d}_overlay.jpg",      overlay_panel)
        cv2.imwrite(f"{OUT_DIR}/frame{i:02d}_combined.jpg",     full_panel)

        s_min, s_max = s_depth_m.min(), s_depth_m.max()
        w_min, w_max = w_depth_m.min(), w_depth_m.max()
        print(f"  frame {i:02d} → side [{s_min:.2f}, {s_max:.2f}]m  "
              f"wrist [{w_min:.2f}, {w_max:.2f}]m")

    print(f"\nSaved {N_FRAMES * 3} images to {OUT_DIR}/")
    print("  frame*_depth_only.jpg  — TURBO depth colourmap, no RGB")
    print("  frame*_overlay.jpg     — 50/50 depth + RGB blend")
    print("  frame*_combined.jpg    — both rows stacked vertically")


if __name__ == "__main__":
    main()
