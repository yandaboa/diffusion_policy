"""
Benchmark DA3METRIC-LARGE inference latency using frames from side + wrist
camera videos (depth_testing/videos/0/1.mp4  and  .../0/2.mp4).

Runs on CUDA device 0.  Reports:
  - Model load time
  - Warmup latency (first few frames, JIT/CUDA compile overhead)
  - Steady-state latency per frame under three strategies:
      A) 2-frame batch  (side + wrist together in one model.inference call)
      B) Serial         (side first, then wrist — two separate calls)
      C) Single frame   (wrist only — cheapest possible)

Usage:
    conda run -n env_uwlab python3 bench_da3_latency.py
"""

import sys
import time
import numpy as np
import cv2
import torch

sys.path.insert(0, "/home/yandabao/diffusion_policy/Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3

SIDE_VIDEO  = "depth_testing/videos/0/1.mp4"   # D435
WRIST_VIDEO = "depth_testing/videos/0/2.mp4"   # D415

MODEL_ID    = "depth-anything/DA3METRIC-LARGE"
DEVICE      = torch.device("cuda:0")

N_WARMUP    = 5
N_BENCH     = 50    # frames to time in steady-state


# ── helpers ──────────────────────────────────────────────────────────────────

def read_frames(path, n):
    """Read n evenly-spaced RGB frames from a video file as list of np.ndarray."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def timed(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_ID} …")
    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(MODEL_ID).to(device=DEVICE)
    model.eval()
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s\n")

    n_needed = N_WARMUP + N_BENCH
    print(f"Reading {n_needed} frames from each video …")
    side_frames  = read_frames(SIDE_VIDEO,  n_needed)
    wrist_frames = read_frames(WRIST_VIDEO, n_needed)
    print(f"  side  frames: {len(side_frames)}  shape={side_frames[0].shape}")
    print(f"  wrist frames: {len(wrist_frames)}  shape={wrist_frames[0].shape}\n")

    # ── Strategy A: 2-frame batch ────────────────────────────────────────────
    print("── Strategy A: 2-frame batch (side + wrist together) ──────────────")
    print(f"  Warming up ({N_WARMUP} frames) …")
    for i in range(N_WARMUP):
        model.inference([side_frames[i], wrist_frames[i]])
    torch.cuda.synchronize()

    latencies_a = []
    for i in range(N_WARMUP, N_WARMUP + N_BENCH):
        _, dt = timed(model.inference, [side_frames[i], wrist_frames[i]])
        latencies_a.append(dt * 1000)

    la = np.array(latencies_a)
    print(f"  mean={la.mean():.1f}ms  median={np.median(la):.1f}ms  "
          f"p95={np.percentile(la,95):.1f}ms  min={la.min():.1f}ms  max={la.max():.1f}ms")
    print(f"  → achievable rate: {1000/la.mean():.1f} fps (per pair)\n")

    # ── Strategy B: serial (side then wrist) ────────────────────────────────
    print("── Strategy B: serial (two separate calls per frame) ───────────────")
    print(f"  Warming up ({N_WARMUP} frames) …")
    for i in range(N_WARMUP):
        model.inference([side_frames[i]])
        model.inference([wrist_frames[i]])
    torch.cuda.synchronize()

    latencies_b = []
    for i in range(N_WARMUP, N_WARMUP + N_BENCH):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.inference([side_frames[i]])
        model.inference([wrist_frames[i]])
        torch.cuda.synchronize()
        latencies_b.append((time.perf_counter() - t0) * 1000)

    lb = np.array(latencies_b)
    print(f"  mean={lb.mean():.1f}ms  median={np.median(lb):.1f}ms  "
          f"p95={np.percentile(lb,95):.1f}ms  min={lb.min():.1f}ms  max={lb.max():.1f}ms")
    print(f"  → achievable rate: {1000/lb.mean():.1f} fps (per pair)\n")

    # ── Strategy C: single frame (wrist only) ───────────────────────────────
    print("── Strategy C: single frame (wrist only) ──────────────────────────")
    latencies_c = []
    for i in range(N_WARMUP, N_WARMUP + N_BENCH):
        _, dt = timed(model.inference, [wrist_frames[i]])
        latencies_c.append(dt * 1000)

    lc = np.array(latencies_c)
    print(f"  mean={lc.mean():.1f}ms  median={np.median(lc):.1f}ms  "
          f"p95={np.percentile(lc,95):.1f}ms  min={lc.min():.1f}ms  max={lc.max():.1f}ms")
    print(f"  → achievable rate: {1000/lc.mean():.1f} fps (per frame)\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("═══ Summary ════════════════════════════════════════════════════════")
    print(f"  2-frame batch : {la.mean():.0f}ms ({1000/la.mean():.1f} fps)")
    print(f"  Serial 2 calls: {lb.mean():.0f}ms ({1000/lb.mean():.1f} fps)")
    print(f"  Single frame  : {lc.mean():.0f}ms ({1000/lc.mean():.1f} fps)")
    robot_hz = 10
    print(f"\n  Robot control loop target: {robot_hz} Hz → budget {1000/robot_hz:.0f}ms per cycle")
    for label, lat in [("2-frame batch", la.mean()), ("Serial", lb.mean()), ("Single", lc.mean())]:
        ok = "✓ fits" if lat < 1000/robot_hz else "✗ too slow"
        print(f"    {label:15s}: {lat:.0f}ms  {ok}")


if __name__ == "__main__":
    main()
