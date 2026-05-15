#!/usr/bin/env python3
"""Benchmark DA3NESTED-GIANT-LARGE latency for 2 cameras × N history frames.

Run in the DA3 conda env:
    conda run -n DA3 python benchmark_da3_nested.py [--history 3] [--device 0]

Tests three modes:
  1. Monocular (current) — 2 separate single-frame calls, DA3METRIC-LARGE
  2. Multi-view, no pose  — 1 call with 2*N images, DA3NESTED-GIANT-LARGE
  3. Multi-view + pose    — same, but intrinsics+extrinsics provided

Camera setup assumed:
  - Side camera : fixed, 640×480, fx≈fy≈615 px
  - Wrist camera: robot-mounted, same intrinsics, pose varies per frame
"""

import argparse
import sys
import os
import time

import numpy as np

_DA3_SRC = os.path.join(os.path.dirname(__file__), 'Depth-Anything-3', 'src')
sys.path.insert(0, _DA3_SRC)


# ── Realistic dummy camera intrinsics (D435 @ 640×480) ───────────────────────
FX, FY, CX, CY = 615.0, 615.0, 320.0, 240.0

def _K() -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0], K[1, 1] = FX, FY
    K[0, 2], K[1, 2] = CX, CY
    return K


def _random_w2c(rng: np.random.Generator) -> np.ndarray:
    """Random but valid world-to-camera 4×4 matrix (camera ~1 m from origin)."""
    axis  = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, 0.3)          # small rotation
    K_skew = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
    R = (np.eye(3) + np.sin(angle) * K_skew
         + (1 - np.cos(angle)) * K_skew @ K_skew)
    t = np.array([0.0, 0.0, 1.0]) + rng.uniform(-0.05, 0.05, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T.astype(np.float64)


def make_inputs(history: int, rng: np.random.Generator):
    """Return (images, intrinsics, extrinsics) for 2*history views."""
    H, W = 480, 640
    imgs = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            for _ in range(2 * history)]

    # Intrinsics: same K for all views
    intrinsics = np.stack([_K()] * (2 * history))   # (2N, 3, 3)

    # Extrinsics:
    #   views 0..history-1   = side camera (fixed pose, repeated)
    #   views history..2N-1  = wrist camera (new pose each frame)
    side_w2c = _random_w2c(rng)
    extrinsics = np.stack(
        [side_w2c] * history
        + [_random_w2c(rng) for _ in range(history)]
    )   # (2N, 4, 4)

    return imgs, intrinsics, extrinsics


def benchmark(model, history: int, with_pose: bool, n_warmup: int, n_timed: int,
              process_res: int = 504, label_extra: str = ""):
    rng = np.random.default_rng(42)

    print(f"\n{'─'*60}")
    label = (f"history={history}  2×{history}={2*history} views"
             f"  pose={'yes' if with_pose else 'no '}"
             f"  res={process_res}"
             + (f"  {label_extra}" if label_extra else ""))
    print(f"  {label}")
    print(f"{'─'*60}")

    imgs, intrinsics, extrinsics = make_inputs(history, rng)

    kwargs = dict(process_res=process_res)
    if with_pose:
        kwargs['intrinsics'] = intrinsics
        kwargs['extrinsics'] = extrinsics

    # warmup
    for i in range(n_warmup):
        _ = model.inference(imgs, **kwargs)
        print(f"  warmup {i+1}/{n_warmup} done", flush=True)

    # timed
    times = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        pred = model.inference(imgs, **kwargs)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        times.append(elapsed)
        print(f"  run {i+1}/{n_timed}: {elapsed:.1f} ms", flush=True)

    times = np.array(times)
    print(f"\n  mean={times.mean():.1f} ms  std={times.std():.1f} ms"
          f"  min={times.min():.1f} ms  max={times.max():.1f} ms")
    print(f"  → {'FAST ENOUGH' if times.mean() < 100 else 'TOO SLOW'} for 10 Hz"
          f" (budget = 100 ms/frame)")
    print(f"  depth shape: {pred.depth.shape}"
          f"  conf: {'yes' if hasattr(pred, 'conf') and pred.conf is not None else 'no'}")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=int, default=2,
                        help='Number of frames per camera (default: 2)')
    parser.add_argument('--device',  type=int, default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--warmup',  type=int, default=2,
                        help='Warmup iterations (default: 2)')
    parser.add_argument('--runs',    type=int, default=5,
                        help='Timed iterations (default: 5)')
    parser.add_argument('--also_metric', action='store_true',
                        help='Also benchmark DA3METRIC-LARGE (current baseline)')
    parser.add_argument('--compile', action='store_true',
                        help='Try torch.compile on the model (requires PyTorch ≥ 2)')
    parser.add_argument('--res', type=int, nargs='+', default=[504, 378, 252],
                        help='process_res values to sweep (default: 504 378 252)')
    args = parser.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device(f'cuda:{args.device}')

    # ── Baseline: DA3METRIC-LARGE (current setup, 2 separate mono calls) ─────
    if args.also_metric:
        print("\n" + "="*60)
        print("BASELINE: DA3METRIC-LARGE  (2 separate monocular calls)")
        print("="*60)
        metric_model = DepthAnything3.from_pretrained(
            'depth-anything/DA3METRIC-LARGE').to(device=device)
        metric_model.eval()

        rng = np.random.default_rng(42)
        H, W = 480, 640
        side_img  = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        wrist_img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

        print("\n  warmup ...", flush=True)
        for _ in range(args.warmup):
            _ = metric_model.inference([side_img, wrist_img])

        times = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            _ = metric_model.inference([side_img])
            _ = metric_model.inference([wrist_img])
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000
            times.append(elapsed)
            print(f"  run {i+1}/{args.runs}: {elapsed:.1f} ms", flush=True)

        times = np.array(times)
        print(f"\n  mean={times.mean():.1f} ms  std={times.std():.1f} ms")
        del metric_model
        torch.cuda.empty_cache()

    # ── DA3NESTED-GIANT-LARGE ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DA3NESTED-GIANT-LARGE")
    print("="*60)
    print("Loading model (this takes ~30 s) ...", flush=True)
    nested_model = DepthAnything3.from_pretrained(
        'depth-anything/DA3NESTED-GIANT-LARGE').to(device=device)
    nested_model.eval()
    print("Model ready.\n", flush=True)

    # ── Sweep process_res (flash attn + bf16 already active internally) ───────
    print("\n" + "="*60)
    print("RESOLUTION SWEEP  (no pose, eager mode)")
    print("="*60)
    for res in args.res:
        benchmark(nested_model, args.history, with_pose=False,
                  n_warmup=args.warmup, n_timed=args.runs, process_res=res)

    # ── With pose — sweep all resolutions ────────────────────────────────────
    print("\n" + "="*60)
    print("RESOLUTION SWEEP  (with pose, eager mode)")
    print("="*60)
    for res in args.res:
        benchmark(nested_model, args.history, with_pose=True,
                  n_warmup=args.warmup, n_timed=args.runs, process_res=res)

    # ── torch.compile ─────────────────────────────────────────────────────────
    if args.compile:
        import torch
        # Fix 1: allow .item() calls inside compiled regions (fixes RoPE graph break).
        # Without this, dynamo splits the graph at every int(tensor) call.
        torch._dynamo.config.capture_scalar_outputs = True

        for compile_mode in ('default', 'reduce-overhead'):
            print("\n" + "="*60)
            print(f"torch.compile  mode='{compile_mode}'")
            print("  note: reduce-overhead skips cudagraphs if inputs are mutated")
            print("="*60)
            print("Compiling … (first call triggers tracing, will be slow)", flush=True)
            import copy
            compiled_model = copy.copy(nested_model)
            compiled_model.model = torch.compile(
                nested_model.model, mode=compile_mode, fullgraph=False)
            for res in args.res:
                benchmark(compiled_model, args.history, with_pose=False,
                          n_warmup=max(args.warmup, 3), n_timed=args.runs,
                          process_res=res, label_extra=compile_mode)
            benchmark(compiled_model, args.history, with_pose=True,
                      n_warmup=max(args.warmup, 3), n_timed=args.runs,
                      process_res=args.res[0], label_extra=compile_mode)
            del compiled_model
            torch.cuda.empty_cache()

    # ── history=2 reference if we ran a different history ─────────────────────
    if args.history != 2:
        print("\n" + "="*60)
        print("REFERENCE: history=2")
        print("="*60)
        benchmark(nested_model, 2, with_pose=False,
                  n_warmup=1, n_timed=args.runs, process_res=args.res[0])
        benchmark(nested_model, 2, with_pose=True,
                  n_warmup=1, n_timed=args.runs, process_res=args.res[0])


if __name__ == '__main__':
    main()
