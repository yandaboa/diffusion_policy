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
"""

import sys
import os
import argparse
import time
import contextlib
import io

import numpy as np
from multiprocessing.shared_memory import SharedMemory

# Depth-Anything-3 source lives two directories above this file's location
_DA3_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../Depth-Anything-3/src')
)
sys.path.insert(0, _DA3_SRC)

_MAX_FRAME_H = 480
_MAX_FRAME_W = 640
_MAX_DEPTH_H = 512
_MAX_DEPTH_W = 768

# State constants — must match da3_depth_client.py
_IDLE         = 0
_FRAME_READY  = 1
_DEPTH_READY  = 2
_WORKER_READY = 10
_STOP         = 255


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shm_prefix', required=True,
                        help='Shared-memory name prefix created by the client')
    parser.add_argument('--device',   type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--model_id', default='depth-anything/DA3METRIC-LARGE',
                        help='HuggingFace model repo ID')
    args = parser.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device(f'cuda:{args.device}')
    p = args.shm_prefix

    # Attach to shared-memory blocks allocated by the client
    shm = {k: SharedMemory(name=f'{p}_{k}') for k in
           ('state', 'meta', 'sf', 'wf', 'sd', 'wd')}

    state = np.ndarray(1, dtype=np.uint8,  buffer=shm['state'].buf)
    meta  = np.ndarray(4, dtype=np.uint32, buffer=shm['meta'].buf)
    # meta layout: [frame_h, frame_w, depth_h, depth_w]

    print(f'[DA3 worker] Loading {args.model_id} on cuda:{args.device} …',
          flush=True)
    model = DepthAnything3.from_pretrained(args.model_id).to(device=device)
    model.eval()
    print('[DA3 worker] Model ready.', flush=True)

    state[0] = _WORKER_READY  # one-time signal to client

    try:
        while True:
            # ── wait for a frame pair ────────────────────────────────────
            while True:
                s = int(state[0])
                if s == _FRAME_READY:
                    break
                if s == _STOP:
                    print('[DA3 worker] STOP received, exiting.', flush=True)
                    return
                time.sleep(0.001)

            # ── read frames from shared memory ───────────────────────────
            fh, fw = int(meta[0]), int(meta[1])
            side_rgb  = np.ndarray(
                (fh, fw, 3), dtype=np.uint8, buffer=shm['sf'].buf).copy()
            wrist_rgb = np.ndarray(
                (fh, fw, 3), dtype=np.uint8, buffer=shm['wf'].buf).copy()

            # ── run inference (stdout suppressed — DA3 logs timing internally) ─
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pred = model.inference([side_rgb, wrist_rgb])
                side_raw  = np.asarray(pred.depth[0], dtype=np.float32)
                wrist_raw = np.asarray(pred.depth[1], dtype=np.float32)
            except Exception as exc:
                print(f'[DA3 worker] inference error: {exc}', flush=True)
                # leave state as FRAME_READY so client sees a timeout
                continue

            # ── write depth maps to shared memory ────────────────────────
            dh, dw = side_raw.shape
            meta[2], meta[3] = dh, dw

            sd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                            buffer=shm['sd'].buf)
            wd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                            buffer=shm['wd'].buf)
            sd[:dh, :dw] = side_raw
            wd[:dh, :dw] = wrist_raw

            state[0] = _DEPTH_READY  # signal client

    finally:
        for s in shm.values():
            s.close()


if __name__ == '__main__':
    main()
