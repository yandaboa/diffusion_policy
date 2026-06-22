"""Fast-FoundationStereo depth client (mirrors da3_depth_client.py).

Manages a FoundationStereo subprocess running in the FFS conda env, talking over POSIX
shared memory. ``infer(left, right, K, baseline)`` returns a metric depth map (metres) in
the LEFT-IR frame. The worker runs under ``FFS_PYTHON`` (its own env) so this process
needs no FoundationStereo deps.

  from diffusion_policy.real_world.ffs_depth_client import FFSDepthClient
  from diffusion_policy.real_world.realsense_stereo import capture_stereo
  with FFSDepthClient() as ffs:
      f = capture_stereo(serial)
      depth_m = ffs.infer(f.left, f.right, f.K_ir, f.baseline_m)
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np

_FFS_PYTHON = os.environ.get("FFS_PYTHON", "/home/yandabao/miniforge3/envs/foundstereo/bin/python3")
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "ffs_worker.py")
# Fast-FoundationStereo submodule at the diffusion_policy repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FFS_REPO = os.environ.get("FFS_REPO", os.path.join(_REPO_ROOT, "Fast-FoundationStereo"))
_FFS_CKPT = os.environ.get("FFS_CKPT", os.path.join(_FFS_REPO, "weights/23-36-37/model_best_bp2_serialize.pth"))

_IDLE, _FRAME_READY, _DEPTH_READY, _WORKER_READY, _STOP = (
    np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(10), np.uint8(255))

_MAX_H, _MAX_W = 720, 1280  # must match ffs_worker.py


class FFSDepthClient:
    """Drives a FoundationStereo subprocess for stereo->metric-depth inference."""

    def __init__(self, ffs_python: str = _FFS_PYTHON, repo: str = _FFS_REPO, ckpt: str = _FFS_CKPT,
                 device: str = "cuda", iters: int = 8, max_disp: int = 192, mock: bool = False):
        self._python = ffs_python
        self._repo = repo
        self._ckpt = ckpt
        self._device = device
        self._iters = iters
        self._max_disp = max_disp
        self._mock = mock
        self._prefix = f"ffs_{uuid.uuid4().hex[:8]}"
        self._proc: "subprocess.Popen | None" = None
        self._shms: "dict[str, SharedMemory]" = {}
        self._state = None

    def start(self) -> "FFSDepthClient":
        p = self._prefix
        alloc = {
            "state": 1, "meta": 6 * 4, "params": 5 * 8,
            "left": _MAX_H * _MAX_W, "right": _MAX_H * _MAX_W,
            "depth": _MAX_H * _MAX_W * 4,
        }
        for name, size in alloc.items():
            self._shms[name] = SharedMemory(name=f"{p}_{name}", create=True, size=size)
        self._state = np.ndarray(1, np.uint8, buffer=self._shms["state"].buf)
        self._meta = np.ndarray(6, np.uint32, buffer=self._shms["meta"].buf)
        self._params = np.ndarray(5, np.float64, buffer=self._shms["params"].buf)
        self._left = np.ndarray(_MAX_H * _MAX_W, np.uint8, buffer=self._shms["left"].buf)
        self._right = np.ndarray(_MAX_H * _MAX_W, np.uint8, buffer=self._shms["right"].buf)
        self._depth = np.ndarray(_MAX_H * _MAX_W, np.float32, buffer=self._shms["depth"].buf)
        self._state[0] = _IDLE

        cmd = [self._python, _WORKER_SCRIPT, "--prefix", p, "--device", self._device,
               "--iters", str(self._iters), "--max-disp", str(self._max_disp)]
        cmd += ["--mock"] if self._mock else ["--repo", self._repo, "--ckpt", self._ckpt]
        self._proc = subprocess.Popen(cmd)
        return self

    def wait_ready(self, timeout: float = 120.0) -> None:
        t0 = time.time()
        while int(self._state[0]) != int(_WORKER_READY):
            if self._proc.poll() is not None:
                raise RuntimeError(f"ffs_worker exited early (code {self._proc.returncode})")
            if time.time() - t0 > timeout:
                raise TimeoutError("ffs_worker did not become ready")
            time.sleep(0.01)
        self._state[0] = _IDLE

    def infer(self, left: np.ndarray, right: np.ndarray, K: np.ndarray,
              baseline: float, timeout: float = 10.0) -> np.ndarray:
        """Rectified IR stereo pair -> metric depth (H, W) float32, metres (left-IR frame)."""
        left = np.ascontiguousarray(left, np.uint8)
        right = np.ascontiguousarray(right, np.uint8)
        h, w = left.shape
        assert h <= _MAX_H and w <= _MAX_W, f"frame {h}x{w} exceeds shm {_MAX_H}x{_MAX_W}"
        self._meta[0], self._meta[1] = h, w
        self._params[:] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], baseline]
        self._left[:h * w] = left.reshape(-1)
        self._right[:h * w] = right.reshape(-1)
        self._state[0] = _FRAME_READY

        t0 = time.time()
        while int(self._state[0]) != int(_DEPTH_READY):
            if self._proc.poll() is not None:
                raise RuntimeError(f"ffs_worker died (code {self._proc.returncode})")
            if time.time() - t0 > timeout:
                raise TimeoutError("ffs_worker inference timed out")
            time.sleep(0.001)
        oh, ow = int(self._meta[2]), int(self._meta[3])
        depth = self._depth[:oh * ow].reshape(oh, ow).copy()
        self._state[0] = _IDLE
        return depth

    def stop(self) -> None:
        if self._state is not None:
            self._state[0] = _STOP
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        for s in self._shms.values():
            s.close()
            try:
                s.unlink()
            except FileNotFoundError:
                pass
        self._shms.clear()

    def __enter__(self) -> "FFSDepthClient":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
