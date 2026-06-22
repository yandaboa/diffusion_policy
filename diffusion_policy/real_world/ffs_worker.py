"""Fast-FoundationStereo depth worker (runs in the FFS conda env).

Launched as a subprocess by ``ffs_depth_client.py`` (mirrors ``da3_worker.py``). Reads a
rectified IR stereo pair + (K, baseline) from shared memory, runs FoundationStereo to get
disparity, converts to metric depth, writes it back. State-byte protocol must match the
client.

The FoundationStereo code in ``_load_model`` / ``_infer_depth`` matches the submodule's
``Fast-FoundationStereo/scripts/run_realsense_d455.py`` (serialized full-model checkpoint,
InputPadder, forward(test_mode=True, optimize_build_volume='pytorch1')). ``--mock`` skips
the model (synthetic depth) so the client<->worker handshake can be tested without weights.

Usage (normally invoked by the client, not by hand):
  python ffs_worker.py --prefix ffs_xxxx --repo <FFS_REPO> --ckpt <weights.pth>
  python ffs_worker.py --prefix ffs_xxxx --mock
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# State-byte protocol (must match ffs_depth_client.py)
_IDLE, _FRAME_READY, _DEPTH_READY, _WORKER_READY, _STOP = (
    np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(10), np.uint8(255))

_MAX_H, _MAX_W = 720, 1280  # shared-memory capacity (must match client)


# ── FoundationStereo model (matches Fast-FoundationStereo/scripts/run_realsense_d455.py) ──
def _load_model(repo: str, ckpt: str, device: str = "cuda", valid_iters: int = 8, max_disp: int = 192):
    """Load the serialized FoundationStereo model. The checkpoint is a PICKLED full
    model object (not a state_dict), so the FFS repo must be on sys.path before load."""
    import torch
    sys.path.insert(0, repo)
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.to(device).eval()
    return model


def _infer_depth(model, left, right, K, baseline, device="cuda", iters=8):
    """Rectified IR stereo (H,W uint8) -> metric depth (H,W float32, metres).

    Mirrors run_realsense_d455.infer_disparity: tile IR to 3ch, InputPadder(divis_by=32),
    forward(test_mode=True, optimize_build_volume='pytorch1'), depth = fx*baseline/disp.
    """
    import torch
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE
    l = np.repeat(left[..., None], 3, axis=2)
    r = np.repeat(right[..., None], 3, axis=2)
    H, W = l.shape[:2]
    lt = torch.as_tensor(l).to(device).float()[None].permute(0, 3, 1, 2)
    rt = torch.as_tensor(r).to(device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(lt.shape, divis_by=32, force_square=False)
    lt, rt = padder.pad(lt, rt)
    with torch.inference_mode(), torch.amp.autocast(device, enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(lt, rt, iters=iters, test_mode=True, optimize_build_volume="pytorch1")
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W).clip(0, None)
    depth = np.zeros((H, W), np.float32)
    valid = disp > 0
    depth[valid] = K[0, 0] * baseline / disp[valid]  # fx * baseline / disparity
    return depth


def _mock_depth(left, right, K, baseline):
    """Synthetic depth for plumbing tests: smooth horizontal ramp ~0.4-0.8 m."""
    h, w = left.shape
    ramp = np.linspace(0.4, 0.8, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    return ramp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--repo", default=os.environ.get("FFS_REPO", ""))
    ap.add_argument("--ckpt", default=os.environ.get("FFS_CKPT", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=8)       # FFS valid_iters
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--mock", action="store_true", help="skip FoundationStereo (test mode)")
    args = ap.parse_args()
    p = args.prefix

    shms = {name: SharedMemory(name=f"{p}_{name}") for name in
            ("state", "meta", "params", "left", "right", "depth")}
    state = np.ndarray(1, np.uint8, buffer=shms["state"].buf)
    meta = np.ndarray(6, np.uint32, buffer=shms["meta"].buf)     # [h,w, out_h,out_w, _, _]
    params = np.ndarray(5, np.float64, buffer=shms["params"].buf)  # [fx,fy,ppx,ppy,baseline]
    left_buf = np.ndarray(_MAX_H * _MAX_W, np.uint8, buffer=shms["left"].buf)
    right_buf = np.ndarray(_MAX_H * _MAX_W, np.uint8, buffer=shms["right"].buf)
    depth_buf = np.ndarray(_MAX_H * _MAX_W, np.float32, buffer=shms["depth"].buf)

    model = (None if args.mock else
             _load_model(args.repo, args.ckpt, args.device, args.iters, args.max_disp))
    state[0] = _WORKER_READY
    print(f"[ffs_worker] ready (mock={args.mock})", flush=True)

    try:
        while True:
            if int(state[0]) == int(_STOP):
                break
            if int(state[0]) != int(_FRAME_READY):
                time.sleep(0.001)
                continue
            h, w = int(meta[0]), int(meta[1])
            left = left_buf[:h * w].reshape(h, w).copy()
            right = right_buf[:h * w].reshape(h, w).copy()
            fx, fy, ppx, ppy, baseline = params
            K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1.0]])
            depth = (_mock_depth(left, right, K, baseline) if args.mock
                     else _infer_depth(model, left, right, K, baseline, args.device, args.iters))
            oh, ow = depth.shape
            depth_buf[:oh * ow] = depth.reshape(-1)
            meta[2], meta[3] = oh, ow
            state[0] = _DEPTH_READY
    finally:
        for s in shms.values():
            s.close()


if __name__ == "__main__":
    main()
