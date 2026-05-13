"""Debug script: grab depth frames from the side (D435) and wrist (D415) RealSense
cameras and visualize them using the exact preprocessing pipeline the depth-student
policy sees at inference time.

Preprocessing mirrors depth_dagger_cfg.py:
  DEPTH_CLIP = (0.01, 2.0) m
  output_size = (224, 224)
  normalize to [0, 1]: (depth_m - 0.01) / 1.99

Usage:
    python debug_depth_cameras.py

Controls (OpenCV window):
    Q  — quit
    S  — save current frames to /tmp/depth_debug_<timestamp>.npz
    C  — toggle color overlay alongside depth

Camera serials (matching eval_real_robot.py):
    Side  (D435): 832112070487
    Wrist (D415): 746112060198
"""

import time
import numpy as np
import cv2
import pyrealsense2 as rs

# ── Constants matching the sim ──────────────────────────────────────────────
DEPTH_CLIP = (0.01, 2.0)   # metres, same as depth_dagger_cfg.py
IMG_H, IMG_W = 224, 224    # policy input resolution

# Camera serials from eval_real_robot.py
SIDE_SERIAL  = "832112070487"   # D435
WRIST_SERIAL = "746112060198"   # D415

# Capture resolution and FPS — D435/D415 both support 848×480 @ 30.
# Using a lower res here since we resize to 224×224 anyway.
CAP_W, CAP_H = 640, 480
CAP_FPS = 30


# ── Depth normalization (exactly as process_image() in observations.py) ─────
def process_depth(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw uint16 depth frame to float32 [0,1] at 224×224.

    Args:
        depth_u16: (H, W) uint16 depth from pyrealsense2 (units = depth_scale metres).
        depth_scale: metres per depth unit, from sensor.get_depth_scale().

    Returns:
        (IMG_H, IMG_W) float32 normalized to [0, 1].
    """
    d_min, d_max = DEPTH_CLIP

    depth_m = depth_u16.astype(np.float32) * depth_scale
    # Zeros = no-return pixels → treat as max range (same as nan_to_num in sim)
    depth_m[depth_m == 0.0] = d_max
    np.clip(depth_m, d_min, d_max, out=depth_m)
    depth_norm = (depth_m - d_min) / (d_max - d_min)

    # Bilinear resize to match the policy's input resolution
    depth_resized = cv2.resize(depth_norm, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return depth_resized


def norm_to_vis(depth_norm: np.ndarray) -> np.ndarray:
    """Float32 [0,1] → BGR uint8 via TURBO colormap for display.

    TURBO (not JET) keeps near-zero values visually distinct from pure black,
    making no-return pixels (mapped to 0) easy to spot as dark purple rather
    than blending into the background.
    """
    gray = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def start_camera(serial: str) -> tuple:
    """Start a single RealSense pipeline (depth + colour). Returns (pipeline, align, depth_scale).

    Each camera gets its own rs.align instance — sharing one across pipelines
    causes corrupt frames on the second camera.
    """
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, CAP_W, CAP_H, rs.format.z16, CAP_FPS)
    cfg.enable_stream(rs.stream.color, CAP_W, CAP_H, rs.format.bgr8, CAP_FPS)

    pipeline = rs.pipeline()
    profile  = pipeline.start(cfg)

    # Enable global timestamp (accurate wall-clock sync)
    dev = profile.get_device()
    dev.first_color_sensor().set_option(rs.option.global_time_enabled, 1)

    depth_scale = dev.first_depth_sensor().get_depth_scale()
    align       = rs.align(rs.stream.color)
    print(f"  [{serial}] started  depth_scale={depth_scale:.6f} m/unit")
    return pipeline, align, depth_scale


def grab(pipeline, align, scale, label):
    """Grab one aligned frameset. Returns (depth_norm 224×224, color CAP_H×CAP_W×3)."""
    try:
        frames     = pipeline.wait_for_frames(timeout_ms=500)
        frames     = align.process(frames)
        df         = frames.get_depth_frame()
        cf         = frames.get_color_frame()
        if not df:
            print(f"[WARN] {label}: no depth frame")
            return np.zeros((IMG_H, IMG_W), np.float32), np.zeros((CAP_H, CAP_W, 3), np.uint8)
        depth_norm = process_depth(np.asarray(df.get_data()), scale)
        color      = np.asarray(cf.get_data()).copy() if cf else np.zeros((CAP_H, CAP_W, 3), np.uint8)
        return depth_norm, color
    except Exception as e:
        print(f"[WARN] {label}: frame grab failed — {e}")
        return np.zeros((IMG_H, IMG_W), np.float32), np.zeros((CAP_H, CAP_W, 3), np.uint8)


def main():
    print("Starting depth debug — connecting to cameras…")
    print(f"  Side  serial: {SIDE_SERIAL}")
    print(f"  Wrist serial: {WRIST_SERIAL}")

    side_pipeline = side_align = side_scale = None
    wrist_pipeline = wrist_align = wrist_scale = None

    try:
        side_pipeline, side_align, side_scale = start_camera(SIDE_SERIAL)
    except Exception as e:
        print(f"[ERROR] Could not open side camera ({SIDE_SERIAL}): {e}")

    try:
        wrist_pipeline, wrist_align, wrist_scale = start_camera(WRIST_SERIAL)
    except Exception as e:
        print(f"[ERROR] Could not open wrist camera ({WRIST_SERIAL}): {e}")

    if side_pipeline is None and wrist_pipeline is None:
        print("[FATAL] No cameras available. Check USB connections and serial numbers.")
        return

    cv2.namedWindow("Depth Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Debug", IMG_W * 2, IMG_H + 20)

    print("\nReady! Press Q to quit, S to save .npz, C to toggle colour overlay.")
    print("Stats printed every 30 frames.\n")

    show_color = False
    frame_idx  = 0
    t_start    = time.monotonic()

    while True:
        # ── Grab frames — each camera has its own pipeline + align ─────────
        if side_pipeline is not None:
            side_norm, side_color = grab(side_pipeline, side_align, side_scale, "side")
        else:
            side_norm  = np.zeros((IMG_H, IMG_W), np.float32)
            side_color = np.zeros((CAP_H, CAP_W, 3), np.uint8)

        if wrist_pipeline is not None:
            wrist_norm, wrist_color = grab(wrist_pipeline, wrist_align, wrist_scale, "wrist")
        else:
            wrist_norm  = np.zeros((IMG_H, IMG_W), np.float32)
            wrist_color = np.zeros((CAP_H, CAP_W, 3), np.uint8)

        # ── Visualization ──────────────────────────────────────────────────
        side_vis  = norm_to_vis(side_norm)
        wrist_vis = norm_to_vis(wrist_norm)

        if show_color:
            # Resize colour to 224×224 and blend over the depth colourmap
            sc_r = cv2.resize(side_color,  (IMG_W, IMG_H))
            wc_r = cv2.resize(wrist_color, (IMG_W, IMG_H))
            side_vis  = cv2.addWeighted(side_vis,  0.5, sc_r, 0.5, 0)
            wrist_vis = cv2.addWeighted(wrist_vis, 0.5, wc_r, 0.5, 0)

        # Annotate
        def annotate(img, label, norm):
            d_m = norm * (DEPTH_CLIP[1] - DEPTH_CLIP[0]) + DEPTH_CLIP[0]
            valid = norm < 0.999  # exclude max-range (no-return) pixels
            mean_m = float(d_m[valid].mean()) if valid.any() else float("nan")
            cv2.putText(img, label,       (5, 18),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(img, f"mean={mean_m:.3f}m", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1)
            return img

        side_vis  = annotate(side_vis,  "SIDE  (D435)", side_norm)
        wrist_vis = annotate(wrist_vis, "WRIST (D415)", wrist_norm)

        # Concatenate side-by-side
        panel = np.concatenate([side_vis, wrist_vis], axis=1)
        cv2.imshow("Depth Debug", panel)

        # ── Console stats every 30 frames ─────────────────────────────────
        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.monotonic() - t_start
            fps = frame_idx / elapsed
            def stats(norm, name):
                d_m = norm * (DEPTH_CLIP[1] - DEPTH_CLIP[0]) + DEPTH_CLIP[0]
                valid = norm < 0.999
                mn = float(d_m[valid].min())   if valid.any() else float("nan")
                mx = float(d_m[valid].max())   if valid.any() else float("nan")
                me = float(d_m[valid].mean())  if valid.any() else float("nan")
                pct_invalid = 100.0 * (~valid).mean()
                print(f"  {name}: min={mn:.3f}m  max={mx:.3f}m  mean={me:.3f}m  "
                      f"no-return={pct_invalid:.1f}%  "
                      f"norm-range=[{norm.min():.3f},{norm.max():.3f}]")
            print(f"[frame {frame_idx}  {fps:.1f} fps]")
            if side_norm  is not None: stats(side_norm,  "side ")
            if wrist_norm is not None: stats(wrist_norm, "wrist")

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ts = int(time.time())
            path = f"/tmp/depth_debug_{ts}.npz"
            np.savez(path, side_norm=side_norm, wrist_norm=wrist_norm)
            print(f"[S] saved to {path}  (side_norm shape={side_norm.shape}, wrist_norm shape={wrist_norm.shape})")
        elif key == ord('c'):
            show_color = not show_color
            print(f"[C] colour overlay: {'ON' if show_color else 'OFF'}")

    cv2.destroyAllWindows()
    if side_pipeline  is not None: side_pipeline.stop()
    if wrist_pipeline is not None: wrist_pipeline.stop()
    print("Done.")


if __name__ == "__main__":
    main()
