"""DA3 depth inference client.

Manages a DA3 subprocess (running in the 'DA3' conda env) and communicates
with it via POSIX shared memory.  Two model modes are supported:

  DA3METRIC-LARGE       — single-frame monocular metric depth.
                          to_metric() converts focal-normalised output → metres.

  DA3NESTED-GIANT-LARGE — multi-view temporal depth with optional pose
                          conditioning.  History=2 buffers 4 images (side t-1,
                          wrist t-1, side t, wrist t) for better accuracy.
                          Output is already metric (metres) — to_metric() is
                          a no-op and can be skipped.

Typical usage (NESTED with pose conditioning)
---------------------------------------------
    with DA3DepthClient(device=0,
                        model_id='depth-anything/DA3NESTED-GIANT-LARGE',
                        process_res=378) as da3:
        da3.wait_ready()
        da3.set_intrinsics(side_K, wrist_K)   # call once after cameras init

        # in the control loop (via DA3DepthBackground):
        bg.put_frame(side_rgb, wrist_rgb,
                     side_extrinsic=side_w2c,   # 4×4 world-to-cam
                     wrist_extrinsic=wrist_w2c)
        ...
        result = bg.get_latest_da3()   # (side_m, wrist_m, timestamp)
"""

import os
import queue as _queue
import subprocess
import threading
import time
import uuid
import warnings

import cv2
import numpy as np
from multiprocessing.shared_memory import SharedMemory

_DA3_PYTHON = os.environ.get(
    'DA3_PYTHON',
    '/home/yandabao/miniforge3/envs/DA3/bin/python3',
)
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), 'da3_worker.py')

# State-byte protocol (must match da3_worker.py)
_IDLE         = np.uint8(0)
_FRAME_READY  = np.uint8(1)
_DEPTH_READY  = np.uint8(2)
_WORKER_READY = np.uint8(10)
_STOP         = np.uint8(255)

_MAX_FRAME_H = 480
_MAX_FRAME_W = 640
_MAX_DEPTH_H = 512
_MAX_DEPTH_W = 768

_METRIC_MODELS = frozenset({'depth-anything/DA3METRIC-LARGE'})
_NESTED_MODELS = frozenset({'depth-anything/DA3NESTED-GIANT-LARGE'})
_ALL_MODELS    = _METRIC_MODELS | _NESTED_MODELS

# Default processing resolution for the NESTED model (fits in 100 ms budget at
# history=2 pose-conditioned on an RTX 4090).  Use 504 if latency is acceptable.
_NESTED_DEFAULT_PROCESS_RES = 378


class DA3DepthClient:
    """Manages a DA3 subprocess for real-time depth inference."""

    def __init__(
        self,
        device:      int = 0,
        model_id:    str = 'depth-anything/DA3METRIC-LARGE',
        process_res: 'int | None' = None,
        da3_python:  str = _DA3_PYTHON,
    ) -> None:
        if model_id not in _ALL_MODELS:
            raise ValueError(
                f"'{model_id}' is not supported. "
                f"Supported models: {sorted(_ALL_MODELS)}"
            )
        self.device      = device
        self.model_id    = model_id
        self.is_nested   = model_id in _NESTED_MODELS
        self._python     = da3_python
        self._prefix     = f'da3_{uuid.uuid4().hex[:8]}'
        self._history    = 2 if self.is_nested else 1
        self._process_res = (
            process_res if process_res is not None
            else (_NESTED_DEFAULT_PROCESS_RES if self.is_nested else 0)
        )

        self._proc: 'subprocess.Popen | None' = None
        self._shms: 'dict[str, SharedMemory]'  = {}
        self._state: 'np.ndarray | None'        = None  # (1,)  uint8
        self._meta:  'np.ndarray | None'        = None  # (8,)  uint32
        self._cp:    'np.ndarray | None'        = None  # (50,) float64, NESTED only

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> 'DA3DepthClient':
        p = self._prefix
        alloc = {
            f'{p}_state': 1,
            f'{p}_meta':  8 * 4,                             # 8 × uint32
            f'{p}_sf':    _MAX_FRAME_H * _MAX_FRAME_W * 3,
            f'{p}_wf':    _MAX_FRAME_H * _MAX_FRAME_W * 3,
            f'{p}_sd':    _MAX_DEPTH_H * _MAX_DEPTH_W * 4,
            f'{p}_wd':    _MAX_DEPTH_H * _MAX_DEPTH_W * 4,
        }
        if self.is_nested:
            alloc[f'{p}_cp'] = 50 * 8  # 50 × float64

        for name, size in alloc.items():
            self._shms[name] = SharedMemory(name=name, create=True, size=size)

        self._state = np.ndarray(1, dtype=np.uint8,  buffer=self._shms[f'{p}_state'].buf)
        self._meta  = np.ndarray(8, dtype=np.uint32, buffer=self._shms[f'{p}_meta'].buf)
        self._state[0] = _IDLE
        self._meta[:]  = 0

        if self.is_nested:
            self._cp = np.ndarray(50, dtype=np.float64,
                                  buffer=self._shms[f'{p}_cp'].buf)
            self._cp[:] = 0.0

        cmd = [
            self._python, _WORKER_SCRIPT,
            '--shm_prefix', self._prefix,
            '--device',     str(self.device),
            '--model_id',   self.model_id,
        ]
        if self.is_nested:
            cmd += ['--history', str(self._history),
                    '--process_res', str(self._process_res)]

        self._proc = subprocess.Popen(cmd)
        return self

    def wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while int(self._state[0]) != int(_WORKER_READY):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f'DA3 worker did not become ready within {timeout:.0f} s')
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f'DA3 worker exited early (rc={self._proc.returncode})')
            time.sleep(0.1)
        self._state[0] = _IDLE

    def stop(self) -> None:
        if self._state is not None:
            try:
                self._state[0] = _STOP
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        for shm in self._shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shms.clear()

    def __enter__(self) -> 'DA3DepthClient':
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── camera parameter setup (NESTED only) ──────────────────────────────

    def set_intrinsics(self, side_K: np.ndarray, wrist_K: np.ndarray) -> None:
        """Store camera intrinsic matrices in shared memory (call once after init).

        Args:
            side_K:  (3, 3) float64 intrinsics for the side camera.
            wrist_K: (3, 3) float64 intrinsics for the wrist camera.
        """
        if not self.is_nested or self._cp is None:
            return
        self._cp[0:9]  = np.asarray(side_K,  dtype=np.float64).flatten()
        self._cp[9:18] = np.asarray(wrist_K, dtype=np.float64).flatten()

    # ── synchronous inference ─────────────────────────────────────────────

    def infer(
        self,
        side_rgb:        np.ndarray,
        wrist_rgb:       np.ndarray,
        side_extrinsic:  'np.ndarray | None' = None,
        wrist_extrinsic: 'np.ndarray | None' = None,
        timeout:         float = 10.0,
    ) -> 'tuple[np.ndarray, np.ndarray]':
        """Synchronous inference — blocks until depth maps are returned.

        Returns:
            (side_depth, wrist_depth): For METRIC, focal-normalised float32
            (pass to to_metric()).  For NESTED, already in metres.
        """
        self.submit(side_rgb, wrist_rgb, side_extrinsic, wrist_extrinsic)
        return self.collect(timeout=timeout)

    # ── async inference ────────────────────────────────────────────────────

    def submit(
        self,
        side_rgb:        np.ndarray,
        wrist_rgb:       np.ndarray,
        side_extrinsic:  'np.ndarray | None' = None,
        wrist_extrinsic: 'np.ndarray | None' = None,
    ) -> None:
        """Write frames (+ optional extrinsics) to shared memory and signal worker."""
        p = self._prefix
        side_rgb  = _to_uint8_rgb(side_rgb)
        wrist_rgb = _to_uint8_rgb(wrist_rgb)

        h, w = side_rgb.shape[:2]
        assert h <= _MAX_FRAME_H and w <= _MAX_FRAME_W

        self._meta[0], self._meta[1] = h, w

        sf = np.ndarray((_MAX_FRAME_H, _MAX_FRAME_W, 3), dtype=np.uint8,
                        buffer=self._shms[f'{p}_sf'].buf)
        wf = np.ndarray((_MAX_FRAME_H, _MAX_FRAME_W, 3), dtype=np.uint8,
                        buffer=self._shms[f'{p}_wf'].buf)
        sf[:h, :w]  = side_rgb
        wf[:h, :w]  = wrist_rgb

        if self.is_nested and self._cp is not None:
            use_pose = (side_extrinsic is not None and wrist_extrinsic is not None)
            self._meta[4] = 1 if use_pose else 0
            if use_pose:
                self._cp[18:34] = np.asarray(side_extrinsic,  dtype=np.float64).flatten()
                self._cp[34:50] = np.asarray(wrist_extrinsic, dtype=np.float64).flatten()
        else:
            self._meta[4] = 0

        self._state[0] = _FRAME_READY

    def collect(self, timeout: float = 10.0) -> 'tuple[np.ndarray, np.ndarray]':
        """Wait for depth maps from the last submit() and return them."""
        p = self._prefix
        deadline = time.monotonic() + timeout
        while int(self._state[0]) != int(_DEPTH_READY):
            if time.monotonic() > deadline:
                raise TimeoutError('DA3 worker did not return depth in time')
            time.sleep(0.001)

        dh, dw = int(self._meta[2]), int(self._meta[3])
        sd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                        buffer=self._shms[f'{p}_sd'].buf)
        wd = np.ndarray((_MAX_DEPTH_H, _MAX_DEPTH_W), dtype=np.float32,
                        buffer=self._shms[f'{p}_wd'].buf)
        side_raw  = sd[:dh, :dw].copy()
        wrist_raw = wd[:dh, :dw].copy()

        self._state[0] = _IDLE
        return side_raw, wrist_raw

    # ── static utilities ──────────────────────────────────────────────────

    @staticmethod
    def to_metric(raw_depth: np.ndarray, focal_px: float) -> np.ndarray:
        """Convert DA3METRIC-LARGE output to metres.

        For DA3NESTED-GIANT-LARGE the output is already metric — this method
        is a no-op and should not be called (the background thread skips it
        automatically when is_nested=True).

        Formula: metric_m = focal_px × raw / 300.0
        """
        assert np.issubdtype(raw_depth.dtype, np.floating)
        return (focal_px * np.asarray(raw_depth, np.float32) / 300.0)

    @staticmethod
    def fuse_with_realsense(
        da3_depth_m:     np.ndarray,
        rs_depth_u16:    np.ndarray,
        depth_scale:     float,
        rs_min_m:        float = 0.05,
        rs_max_m:        float = 4.0,
        min_overlap_px:  int   = 50,
        rs_blend_alpha:  float = 0.9,
        bilateral_sigma: float = 0.05,
    ) -> np.ndarray:
        """Align DA3 metric depth to RealSense, inpaint holes, and smooth.

        Works with both METRIC and NESTED models (both produce metric depth).

        Algorithm:
            1. Convert RS uint16 → metres; resize to DA3 resolution.
            2. valid = RS reading is in range AND DA3 is finite + positive.
            3. scale = median(rs_valid / da3_valid)  — robust to outliers.
            4. Blend: fused = alpha*rs + (1-alpha)*da3_scaled at valid px.
            5. Fill holes with da3_scaled.
            6. Bilateral filter for edge-preserving smoothing.
        """
        assert np.issubdtype(da3_depth_m.dtype, np.floating), \
            "da3_depth_m must be float32 — call to_metric() for METRIC models."
        assert np.issubdtype(rs_depth_u16.dtype, np.unsignedinteger)

        H, W = da3_depth_m.shape
        rs_m   = rs_depth_u16.astype(np.float32) * depth_scale
        rs_m_r = cv2.resize(rs_m, (W, H), interpolation=cv2.INTER_NEAREST)

        rs_valid  = (rs_m_r >= rs_min_m) & (rs_m_r <= rs_max_m)
        da3_valid = np.isfinite(da3_depth_m) & (da3_depth_m > 0.0)
        valid     = rs_valid & da3_valid

        if int(valid.sum()) < min_overlap_px:
            warnings.warn(
                f"DA3 fusion: only {valid.sum()} valid overlap pixels "
                f"(need ≥ {min_overlap_px}). Returning unscaled DA3 depth.",
                stacklevel=2,
            )
            return da3_depth_m.astype(np.float32)

        scale      = float(np.median(rs_m_r[valid] / da3_depth_m[valid]))
        da3_scaled = (da3_depth_m * scale).astype(np.float32)

        fused = da3_scaled.copy()
        fused[rs_valid] = (rs_blend_alpha       * rs_m_r[rs_valid]
                           + (1.0 - rs_blend_alpha) * da3_scaled[rs_valid])

        if bilateral_sigma > 0.0:
            sigma_px = max(1.0, bilateral_sigma * 100.0)
            fused = cv2.bilateralFilter(
                fused, d=5,
                sigmaColor=bilateral_sigma,
                sigmaSpace=sigma_px,
            )

        return fused


class DA3DepthBackground:
    """Runs DA3 inference in a background thread, storing the latest result.

    A single background thread pulls RGB frame pairs (+ optional extrinsics for
    NESTED pose conditioning) from a single-slot queue, calls infer(), converts
    to metric depth if needed, and writes the result to a shared slot.

    Usage
    -----
        bg = DA3DepthBackground(da3_client, side_focal_px, wrist_focal_px)
        bg.start()

        # in control loop:
        bg.put_frame(side_rgb, wrist_rgb,
                     side_extrinsic=side_w2c,    # optional, NESTED only
                     wrist_extrinsic=wrist_w2c)
        ...
        result = bg.get_latest_da3()   # (side_m, wrist_m, timestamp) or None
    """

    def __init__(
        self,
        da3_client:     'DA3DepthClient',
        side_focal_px:  float,
        wrist_focal_px: float,
    ) -> None:
        self._da3            = da3_client
        self._side_focal_px  = side_focal_px
        self._wrist_focal_px = wrist_focal_px
        self._queue: _queue.Queue = _queue.Queue(maxsize=1)
        self._lock   = threading.Lock()
        self._latest: 'tuple[np.ndarray, np.ndarray, float] | None' = None
        self._stop   = threading.Event()
        self._thread: 'threading.Thread | None' = None

    def start(self) -> 'DA3DepthBackground':
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name='DA3DepthBackground')
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> 'DA3DepthBackground':
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            side_rgb, wrist_rgb, side_E, wrist_E = item
            try:
                side_raw, wrist_raw = self._da3.infer(
                    side_rgb, wrist_rgb,
                    side_extrinsic=side_E,
                    wrist_extrinsic=wrist_E,
                )
                # NESTED outputs metric depth directly; METRIC needs conversion
                if self._da3.is_nested:
                    side_m, wrist_m = side_raw, wrist_raw
                else:
                    side_m  = DA3DepthClient.to_metric(side_raw,  self._side_focal_px)
                    wrist_m = DA3DepthClient.to_metric(wrist_raw, self._wrist_focal_px)
                with self._lock:
                    self._latest = (side_m, wrist_m, time.time())
            except Exception as exc:
                print(f'[DA3DepthBackground] inference error: {exc}')

    def put_frame(
        self,
        side_rgb:        np.ndarray,
        wrist_rgb:       np.ndarray,
        side_extrinsic:  'np.ndarray | None' = None,
        wrist_extrinsic: 'np.ndarray | None' = None,
    ) -> None:
        """Submit the latest RGB frames (and optional extrinsics) for inference.

        Drops the previously queued frame if it hasn't been processed yet.
        """
        item = (side_rgb, wrist_rgb, side_extrinsic, wrist_extrinsic)
        try:
            self._queue.get_nowait()
        except _queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except _queue.Full:
            pass

    def get_latest_da3(self) -> 'tuple[np.ndarray, np.ndarray, float] | None':
        """Return (side_m, wrist_m, timestamp) from the most recent inference."""
        with self._lock:
            return self._latest


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)
