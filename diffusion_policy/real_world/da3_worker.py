#!/usr/bin/env python3
"""DA3 inference worker — launched as a subprocess in the DA3 conda env.

Loads the model once on startup, then serves one inference request at a time
via POSIX shared memory.

State-byte protocol (shared with da3_depth_client.py):
    0   IDLE          – waiting for a frame pair
    1   FRAME_READY   – client wrote frames; worker should infer
    2   DEPTH_READY   – worker wrote depth maps; client should read
   10   WORKER_READY  – one-time: model finished loading, ready to serve
  255   STOP          – client requests clean shutdown

Meta array (8 × uint32):
    [0] frame_h   [1] frame_w   (written by client)
    [2] depth_h   [3] depth_w   (written by worker)
    [4] use_pose  (0/1, written by client each frame, NESTED only)
    [5–7] reserved

Cam-params block (cp, NESTED only, 50 × float64):
    [0:9]   side_K  (3×3 intrinsics, flat)
    [9:18]  wrist_K (3×3 intrinsics, flat)
    [18:34] side_E  (4×4 world-to-cam, flat)  — updated per frame
    [34:50] wrist_E (4×4 world-to-cam, flat)  — updated per frame
"""

import sys
import os
import argparse
import collections
import contextlib
import io
import time

import numpy as np
from multiprocessing.shared_memory import SharedMemory

_DA3_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../Depth-Anything-3/src')
)
sys.path.insert(0, _DA3_SRC)

_MAX_FRAME_H = 480
_MAX_FRAME_W = 640
_MAX_DEPTH_H = 512
_MAX_DEPTH_W = 768

_IDLE         = 0
_FRAME_READY  = 1
_DEPTH_READY  = 2
_WORKER_READY = 10
_STOP         = 255


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shm_prefix', required=True)
    parser.add_argument('--device',      type=int, default=0)
    parser.add_argument('--model_id',    default='depth-anything/DA3METRIC-LARGE')
    parser.add_argument('--history',     type=int, default=1,
                        help='Number of frame pairs to buffer (NESTED only)')
    parser.add_argument('--process_res', type=int, default=0,
                        help='DA3 processing resolution (0 = model default)')
    args = parser.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device(f'cuda:{args.device}')
    p = args.shm_prefix
    is_nested = 'NESTED' in args.model_id

    # Attach to shared-memory blocks allocated by the client
    shm_keys = ['state', 'meta', 'sf', 'wf', 'sd', 'wd']
    if is_nested:
        shm_keys.append('cp')
    shm = {k: SharedMemory(name=f'{p}_{k}') for k in shm_keys}

    state = np.ndarray(1, dtype=np.uint8,  buffer=shm['state'].buf)
    meta  = np.ndarray(8, dtype=np.uint32, buffer=shm['meta'].buf)

    cp_buf = None
    if is_nested:
        cp_buf = np.ndarray(50, dtype=np.float64, buffer=shm['cp'].buf)

    process_res = args.process_res or (378 if is_nested else None)

    print(f'[DA3 worker] Loading {args.model_id} on cuda:{args.device} …', flush=True)
    model = DepthAnything3.from_pretrained(args.model_id).to(device=device)
    model.eval()
    print('[DA3 worker] Model ready.', flush=True)

    state[0] = _WORKER_READY

    # Internal history buffers for NESTED mode
    frame_history: collections.deque = collections.deque(maxlen=args.history)
    cam_history:   collections.deque = collections.deque(maxlen=args.history)

    try:
        while True:
            # wait for a frame pair
            while True:
                s = int(state[0])
                if s == _FRAME_READY:
                    break
                if s == _STOP:
                    print('[DA3 worker] STOP received, exiting.', flush=True)
                    return
                time.sleep(0.001)

            # read frames from shared memory
            fh, fw = int(meta[0]), int(meta[1])
            side_rgb  = np.ndarray(
                (fh, fw, 3), dtype=np.uint8, buffer=shm['sf'].buf).copy()
            wrist_rgb = np.ndarray(
                (fh, fw, 3), dtype=np.uint8, buffer=shm['wf'].buf).copy()

            try:
                if is_nested:
                    side_raw, wrist_raw = _infer_nested(
                        model, side_rgb, wrist_rgb,
                        frame_history, cam_history,
                        cp_buf, meta, process_res,
                    )
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        pred = model.inference([side_rgb, wrist_rgb])
                    side_raw  = np.asarray(pred.depth[0], dtype=np.float32)
                    wrist_raw = np.asarray(pred.depth[1], dtype=np.float32)
            except Exception as exc:
                print(f'[DA3 worker] inference error: {exc}', flush=True)
                import traceback
                traceback.print_exc()
                continue

            # write depth maps to shared memory
            dh, dw = side_raw.shape
            meta[2], meta[3] = dh, dw

            sd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                            buffer=shm['sd'].buf)
            wd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                            buffer=shm['wd'].buf)
            sd[:dh, :dw] = side_raw
            wd[:dh, :dw] = wrist_raw

            state[0] = _DEPTH_READY

    finally:
        for s in shm.values():
            s.close()


def _infer_nested(
    model,
    side_rgb:      np.ndarray,
    wrist_rgb:     np.ndarray,
    frame_history: collections.deque,
    cam_history:   collections.deque,
    cp_buf:        'np.ndarray | None',
    meta:          np.ndarray,
    process_res:   'int | None',
) -> 'tuple[np.ndarray, np.ndarray]':
    """Run one inference step for DA3NESTED with internal history accumulation.

    Returns (side_depth_m, wrist_depth_m) for the *latest* frame pair.
    Output is already in metric metres — no to_metric() needed.
    """
    frame_history.append((side_rgb, wrist_rgb))

    use_pose = int(meta[4]) == 1 and cp_buf is not None
    if use_pose:
        params = cp_buf.copy()
        cam_history.append((
            params[0:9].reshape(3, 3),   # side_K
            params[9:18].reshape(3, 3),  # wrist_K
            params[18:34].reshape(4, 4), # side_E (world-to-cam)
            params[34:50].reshape(4, 4), # wrist_E (world-to-cam)
        ))

    n = len(frame_history)
    imgs = [img for pair in frame_history for img in pair]  # flatten pairs

    kw: dict = {}
    if process_res:
        kw['process_res'] = process_res
    if use_pose and len(cam_history) == n:
        Ks = [K for entry in cam_history for K in (entry[0], entry[1])]
        Es = [E for entry in cam_history for E in (entry[2], entry[3])]
        kw['intrinsics'] = np.stack(Ks).astype(np.float64)
        kw['extrinsics'] = np.stack(Es).astype(np.float64)

    with contextlib.redirect_stdout(io.StringIO()):
        pred = model.inference(imgs, **kw)

    # Latest frame pair is always at the tail of the flat image list
    latest_side_idx  = 2 * (n - 1)
    latest_wrist_idx = 2 * (n - 1) + 1
    side_raw  = np.asarray(pred.depth[latest_side_idx],  dtype=np.float32)
    wrist_raw = np.asarray(pred.depth[latest_wrist_idx], dtype=np.float32)
    return side_raw, wrist_raw


if __name__ == '__main__':
    main()
