"""FoundationPose 6-DoF object-pose client (mirrors ffs_depth_client.py).

Manages a FoundationPose subprocess running in the ``foundationpose`` conda env, talking over
POSIX shared memory. Give it a color frame + color-aligned metric depth + intrinsics and it
returns the object's 4x4 pose in the CAMERA frame:

    from diffusion_policy.real_world.foundationpose_client import FoundationPoseClient
    with FoundationPoseClient(mesh_path=".../peg.obj") as fp:
        pose_cam = fp.register(color, depth_m, K, mask)   # first frame (needs object mask)
        while ...:
            pose_cam = fp.track(color, depth_m, K)        # subsequent frames

The worker runs under ``FOUNDATIONPOSE_PYTHON`` (its own env) so this process needs no
FoundationPose deps. This class is camera- and object-agnostic: one instance tracks one object
against one mesh; run several instances (e.g. side + orbbec cameras) and merge downstream.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np

_FP_PYTHON = os.environ.get(
    "FOUNDATIONPOSE_PYTHON", "/home/yandabao/miniforge3/envs/foundationpose/bin/python")
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "foundationpose_worker.py")
# FoundationPose submodule at the diffusion_policy repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FP_REPO = os.environ.get("FOUNDATIONPOSE_REPO", os.path.join(_REPO_ROOT, "FoundationPose"))

_IDLE, _FRAME_READY, _POSE_READY, _WORKER_READY, _STOP = (
    np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(10), np.uint8(255))
_MODE_TRACK, _MODE_REGISTER = 0, 1

_MAX_H, _MAX_W = 1200, 1920  # shared-memory capacity (must cover Orbbec 1280x960); match worker


class FoundationPoseClient:
    """Drives a FoundationPose subprocess for RGB-D -> 6-DoF object pose (camera frame)."""

    def __init__(self, mesh_path: str, fp_python: str = _FP_PYTHON, fp_repo: str = _FP_REPO,
                 est_iter: int = 5, track_iter: int = 2, mock: bool = False):
        self._mesh_path = mesh_path
        self._python = fp_python
        self._repo = fp_repo
        self._est_iter = est_iter
        self._track_iter = track_iter
        self._mock = mock
        self._prefix = f"fp_{uuid.uuid4().hex[:8]}"
        self._proc: "subprocess.Popen | None" = None
        self._shms: "dict[str, SharedMemory]" = {}
        self._state = None
        self._registered = False

    def start(self) -> "FoundationPoseClient":
        p = self._prefix
        alloc = {
            "state": 1, "meta": 8 * 4, "params": 4 * 8,
            "rgb": _MAX_H * _MAX_W * 3, "depth": _MAX_H * _MAX_W * 4,
            "mask": _MAX_H * _MAX_W, "pose": 16 * 8,
        }
        for name, size in alloc.items():
            self._shms[name] = SharedMemory(name=f"{p}_{name}", create=True, size=size)
        self._state = np.ndarray(1, np.uint8, buffer=self._shms["state"].buf)
        self._meta = np.ndarray(8, np.uint32, buffer=self._shms["meta"].buf)
        self._params = np.ndarray(4, np.float64, buffer=self._shms["params"].buf)
        self._rgb = np.ndarray(_MAX_H * _MAX_W * 3, np.uint8, buffer=self._shms["rgb"].buf)
        self._depth = np.ndarray(_MAX_H * _MAX_W, np.float32, buffer=self._shms["depth"].buf)
        self._mask = np.ndarray(_MAX_H * _MAX_W, np.uint8, buffer=self._shms["mask"].buf)
        self._pose = np.ndarray(16, np.float64, buffer=self._shms["pose"].buf)
        self._state[0] = _IDLE

        cmd = [self._python, _WORKER_SCRIPT, "--prefix", p,
               "--max-h", str(_MAX_H), "--max-w", str(_MAX_W)]
        if self._mock:
            cmd += ["--mock"]
        else:
            cmd += ["--fp-repo", self._repo, "--mesh", self._mesh_path,
                    "--est-iter", str(self._est_iter), "--track-iter", str(self._track_iter)]
        self._proc = subprocess.Popen(cmd)
        return self

    def wait_ready(self, timeout: float = 180.0) -> None:
        """Block until the worker loads mesh + networks (FoundationPose init is ~seconds)."""
        t0 = time.time()
        while int(self._state[0]) != int(_WORKER_READY):
            if self._proc.poll() is not None:
                raise RuntimeError(f"fp_worker exited early (code {self._proc.returncode})")
            if time.time() - t0 > timeout:
                raise TimeoutError("fp_worker did not become ready")
            time.sleep(0.01)
        self._state[0] = _IDLE

    def _submit(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray,
                mode: int, mask: "np.ndarray | None") -> None:
        color = np.ascontiguousarray(color, np.uint8)
        depth = np.ascontiguousarray(depth, np.float32)
        assert color.shape[:2] == depth.shape, \
            f"color {color.shape[:2]} vs depth {depth.shape} mismatch"
        h, w = depth.shape
        assert h <= _MAX_H and w <= _MAX_W, f"frame {h}x{w} exceeds shm {_MAX_H}x{_MAX_W}"
        self._meta[0], self._meta[1], self._meta[2] = h, w, mode
        self._params[:] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self._rgb[:h * w * 3] = color.reshape(-1)
        self._depth[:h * w] = depth.reshape(-1)
        if mode == _MODE_REGISTER:
            self._mask[:h * w] = np.ascontiguousarray(mask, bool).reshape(-1).astype(np.uint8)
        self._state[0] = _FRAME_READY

    def _collect(self, timeout: float = 30.0) -> np.ndarray:
        t0 = time.time()
        while int(self._state[0]) != int(_POSE_READY):
            if self._proc.poll() is not None:
                raise RuntimeError(f"fp_worker died (code {self._proc.returncode})")
            if time.time() - t0 > timeout:
                raise TimeoutError("fp_worker pose estimation timed out")
            time.sleep(0.001)
        pose = self._pose.reshape(4, 4).copy()
        self._state[0] = _IDLE
        return pose

    def register(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray,
                 mask: np.ndarray, timeout: float = 60.0) -> np.ndarray:
        """First-frame global registration (needs an object mask) -> (4,4) object-in-camera."""
        self._submit(color, depth, K, _MODE_REGISTER, mask)
        pose = self._collect(timeout)
        self._registered = True
        return pose

    def track(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray,
              timeout: float = 30.0) -> np.ndarray:
        """Per-frame tracking refinement -> (4,4) object-in-camera. Call after ``register``."""
        if not self._registered:
            raise RuntimeError("call register(...) once before track(...)")
        self._submit(color, depth, K, _MODE_TRACK, None)
        return self._collect(timeout)

    def estimate(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray,
                 mask: "np.ndarray | None" = None) -> np.ndarray:
        """Convenience: register on the first call (or whenever a ``mask`` is given), else track."""
        if not self._registered or mask is not None:
            if mask is None:
                raise RuntimeError("first estimate() needs a mask to register")
            return self.register(color, depth, K, mask)
        return self.track(color, depth, K)

    @property
    def registered(self) -> bool:
        return self._registered

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

    def __enter__(self) -> "FoundationPoseClient":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
