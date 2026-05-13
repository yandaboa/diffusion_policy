"""DA3 depth inference client.

Manages a DA3METRIC-LARGE subprocess (running in the 'DA3' conda env) and
communicates with it via POSIX shared memory.  The calling process (robodiff_real,
PyTorch 1.12) is fully decoupled from DA3's runtime requirements (PyTorch ≥ 2).

Only DA3METRIC-LARGE is supported because fuse_with_realsense() requires metric
(absolute-scale) depth output.

Typical usage
-------------
Start the client *before* RealEnv so the ~26 s model-load overlaps with robot
and camera initialisation:

    with DA3DepthClient(device=0) as da3:
        # ... start robot env (~60 s) ...
        da3.wait_ready()                        # blocks until model is loaded

        # synchronous: blocks ~43 ms on RTX 4090 for 640×480 side+wrist batch
        side_raw, wrist_raw = da3.infer(side_rgb, wrist_rgb)

        side_m  = DA3DepthClient.to_metric(side_raw,  focal_px=615.0)
        wrist_m = DA3DepthClient.to_metric(wrist_raw, focal_px=610.0)
        fused   = DA3DepthClient.fuse_with_realsense(side_m, rs_u16, depth_scale)

Async pattern (zero extra latency in a 10 Hz loop)
---------------------------------------------------
    da3.submit(side_rgb, wrist_rgb)   # at the top of the loop (~1 ms)
    # ... do other loop work (~50 ms) ...
    side_raw, wrist_raw = da3.collect()  # already done, returns immediately
"""

import os
import subprocess
import time
import uuid
import warnings

import cv2
import numpy as np
from multiprocessing.shared_memory import SharedMemory

# DA3 conda env Python interpreter.  Override with env-var DA3_PYTHON if needed.
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

# Pre-allocated shared-memory dimensions.  Increase if using higher-res cameras.
_MAX_FRAME_H = 480
_MAX_FRAME_W = 640
_MAX_DEPTH_H = 512
_MAX_DEPTH_W = 768

# Only metric models are supported (output is focal-normalised, not relative).
_METRIC_MODELS = frozenset({'depth-anything/DA3METRIC-LARGE'})


class DA3DepthClient:
    """Manages a DA3METRIC-LARGE subprocess for real-time depth inference."""

    def __init__(
        self,
        device: int = 0,
        model_id: str = 'depth-anything/DA3METRIC-LARGE',
        da3_python: str = _DA3_PYTHON,
    ) -> None:
        if model_id not in _METRIC_MODELS:
            raise ValueError(
                f"'{model_id}' is not a supported metric model. "
                f"fuse_with_realsense requires metric output. "
                f"Supported: {sorted(_METRIC_MODELS)}"
            )
        self.device   = device
        self.model_id = model_id
        self._python  = da3_python
        self._prefix  = f'da3_{uuid.uuid4().hex[:8]}'

        self._proc: subprocess.Popen | None = None
        self._shms: dict[str, SharedMemory] = {}
        self._state: np.ndarray | None = None   # (1,) uint8 view
        self._meta:  np.ndarray | None = None   # (4,) uint32 view

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> 'DA3DepthClient':
        """Allocate shared memory and launch the DA3 subprocess (non-blocking).

        Call wait_ready() later to block until the model finishes loading.
        """
        p = self._prefix
        alloc = {
            f'{p}_state': 1,
            f'{p}_meta':  4 * 4,                            # 4 × uint32
            f'{p}_sf':    _MAX_FRAME_H * _MAX_FRAME_W * 3,  # side frame
            f'{p}_wf':    _MAX_FRAME_H * _MAX_FRAME_W * 3,  # wrist frame
            f'{p}_sd':    _MAX_DEPTH_H * _MAX_DEPTH_W * 4,  # side depth f32
            f'{p}_wd':    _MAX_DEPTH_H * _MAX_DEPTH_W * 4,  # wrist depth f32
        }
        for name, size in alloc.items():
            self._shms[name] = SharedMemory(name=name, create=True, size=size)

        self._state = np.ndarray(
            1, dtype=np.uint8,  buffer=self._shms[f'{p}_state'].buf)
        self._meta  = np.ndarray(
            4, dtype=np.uint32, buffer=self._shms[f'{p}_meta'].buf)
        self._state[0] = _IDLE

        self._proc = subprocess.Popen([
            self._python, _WORKER_SCRIPT,
            '--shm_prefix', self._prefix,
            '--device',     str(self.device),
            '--model_id',   self.model_id,
        ])
        return self

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Block until the worker subprocess finishes loading the model.

        Typically ~26 s for DA3METRIC-LARGE on an RTX 4090.
        """
        deadline = time.monotonic() + timeout
        while int(self._state[0]) != int(_WORKER_READY):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f'DA3 worker did not become ready within {timeout:.0f} s')
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f'DA3 worker exited early (rc={self._proc.returncode})')
            time.sleep(0.1)
        self._state[0] = _IDLE   # ACK, reset to idle

    def stop(self) -> None:
        """Signal the worker to exit, then free all shared memory."""
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

    # ── synchronous inference ─────────────────────────────────────────────

    def infer(
        self,
        side_rgb:  np.ndarray,
        wrist_rgb: np.ndarray,
        timeout:   float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronous inference — blocks until depth maps are returned (~43 ms).

        Args:
            side_rgb:  (H, W, 3) uint8 RGB or float32 [0,1], side camera.
            wrist_rgb: (H, W, 3) uint8 RGB or float32 [0,1], wrist camera.
            timeout:   Seconds before TimeoutError (first call may be slower
                       due to CUDA kernel compilation, use ≥ 10 s).

        Returns:
            (side_raw, wrist_raw): focal-normalised float32 depth at model
            output resolution.  Pass to to_metric() to get metres.
        """
        self.submit(side_rgb, wrist_rgb)
        return self.collect(timeout=timeout)

    # ── async inference (submit / collect) ────────────────────────────────

    def submit(self, side_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        """Write frames to shared memory and signal the worker — non-blocking.

        Pair with collect() to retrieve results.  The typical async pattern
        in a 10 Hz control loop:

            da3.submit(side_rgb, wrist_rgb)   # top of loop  (~1 ms)
            ... other loop work (~50 ms) ...
            side_raw, wrist_raw = da3.collect()  # usually returns immediately
        """
        p = self._prefix
        side_rgb  = _to_uint8_rgb(side_rgb)
        wrist_rgb = _to_uint8_rgb(wrist_rgb)

        h, w = side_rgb.shape[:2]
        assert h <= _MAX_FRAME_H and w <= _MAX_FRAME_W, (
            f'Frame {h}×{w} exceeds max {_MAX_FRAME_H}×{_MAX_FRAME_W}. '
            'Increase _MAX_FRAME_H / _MAX_FRAME_W in da3_depth_client.py.'
        )

        self._meta[0], self._meta[1] = h, w

        sf = np.ndarray(
            (_MAX_FRAME_H, _MAX_FRAME_W, 3), dtype=np.uint8,
            buffer=self._shms[f'{p}_sf'].buf)
        wf = np.ndarray(
            (_MAX_FRAME_H, _MAX_FRAME_W, 3), dtype=np.uint8,
            buffer=self._shms[f'{p}_wf'].buf)
        sf[:h, :w]  = side_rgb
        wf[:h, :w]  = wrist_rgb

        self._state[0] = _FRAME_READY

    def collect(self, timeout: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """Wait for depth maps from the last submit() call and return them.

        Returns:
            (side_raw, wrist_raw): focal-normalised float32 depth arrays.
        """
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
        """Convert DA3METRIC-LARGE focal-normalised output to metres.

        Formula (from DA3 docs): metric_m = focal_px × raw / 300.0

        Args:
            raw_depth: float32 array from infer() / collect() (focal-normalised).
            focal_px:  (fx + fy) / 2 in pixels for the capturing camera.
                       Approximate defaults: D435 @ 640×480 → ~615, D415 → ~610.
                       The median rescaling in fuse_with_realsense() absorbs
                       any error from using approximate focal lengths.

        Returns:
            depth_m: float32 array, same shape, in metres.
        """
        assert np.issubdtype(raw_depth.dtype, np.floating), (
            "raw_depth must be float32 — it should come from infer() / collect(). "
            "Only DA3METRIC-LARGE is supported."
        )
        return (focal_px * np.asarray(raw_depth, np.float32) / 300.0)

    @staticmethod
    def fuse_with_realsense(
        da3_depth_m:     np.ndarray,
        rs_depth_u16:    np.ndarray,
        depth_scale:     float,
        rs_min_m:        float = 0.05,
        rs_max_m:        float = 4.0,
        min_overlap_px:  int   = 50,
        rs_blend_alpha:  float = 0.7,
        bilateral_sigma: float = 0.2,
    ) -> np.ndarray:
        """Align DA3 metric depth to RealSense, inpaint holes, and smooth.

        Only valid for metric DA3 models (asserted via dtype).

        Algorithm
        ---------
        1. Convert RS uint16 → metres; resize (nearest-neighbour) to DA3 res.
        2. valid = RS reading is real  AND  DA3 is finite + positive.
        3. scale = median(rs_valid / da3_valid)   [robust to outliers]
        4. da3_scaled = da3_depth_m × scale
        5. Blend at valid pixels:
               fused = alpha * rs  +  (1-alpha) * da3_scaled
           This preserves RS absolute accuracy while pulling noisy pixels
           toward the smoother DA3 surface (your suggestion).
        6. Fill holes (invalid RS) with da3_scaled.
        7. Bilateral filter on the full map: smooths remaining noise while
           preserving depth discontinuities (standard depth-completion step).

        Args:
            da3_depth_m:     (H, W) float32, DA3 depth in metres (from to_metric).
            rs_depth_u16:    (H′, W′) uint16, raw RealSense depth frame.
            depth_scale:     Metres per RS uint16 unit (from get_depth_scale()).
            rs_min_m:        Lower bound (m) for a RS pixel to count as valid.
            rs_max_m:        Upper bound (m) for a RS pixel to count as valid.
            min_overlap_px:  Minimum valid overlap pixels to fit the scale.
                             Falls back to unscaled DA3 with a warning if fewer.
            rs_blend_alpha:  Weight of RS in the blend at valid pixels (0–1).
                             1.0 = pure RS (no smoothing benefit from DA3),
                             0.0 = pure DA3.  Default 0.7 trusts RS accuracy
                             while pulling toward DA3's smoother surface.
            bilateral_sigma: Sigma (metres) for the bilateral depth filter.
                             Controls how aggressively noise is smoothed; also
                             used as spatial sigma in pixels (sigma_m × 100).
                             Set to 0 to skip bilateral filtering.

        Returns:
            fused_depth_m: (H, W) float32, in metres.
        """
        assert np.issubdtype(da3_depth_m.dtype, np.floating), (
            "da3_depth_m must be float32 — call to_metric() before "
            "fuse_with_realsense().  fuse_with_realsense only works with "
            "metric DA3 models."
        )
        assert np.issubdtype(rs_depth_u16.dtype, np.unsignedinteger), (
            "rs_depth_u16 must be uint16 from the RealSense depth frame."
        )

        H, W = da3_depth_m.shape

        rs_m   = rs_depth_u16.astype(np.float32) * depth_scale
        rs_m_r = cv2.resize(rs_m, (W, H), interpolation=cv2.INTER_NEAREST)

        rs_valid  = (rs_m_r >= rs_min_m) & (rs_m_r <= rs_max_m)
        da3_valid = np.isfinite(da3_depth_m) & (da3_depth_m > 0.0)
        valid     = rs_valid & da3_valid

        if int(valid.sum()) < min_overlap_px:
            warnings.warn(
                f"DA3 fusion: only {valid.sum()} valid overlap pixels "
                f"(need ≥ {min_overlap_px}). Returning unscaled DA3 depth. "
                "Check RealSense connection and camera FOV overlap.",
                stacklevel=2,
            )
            return da3_depth_m.astype(np.float32)

        scale      = float(np.median(rs_m_r[valid] / da3_depth_m[valid]))
        da3_scaled = (da3_depth_m * scale).astype(np.float32)

        # Start from DA3 (covers holes), then overwrite with blended values
        # where RS has a real reading.
        fused = da3_scaled.copy()
        fused[rs_valid] = (rs_blend_alpha       * rs_m_r[rs_valid]
                           + (1.0 - rs_blend_alpha) * da3_scaled[rs_valid])

        # Bilateral filter: smooths noise in flat regions, preserves depth edges.
        # sigmaColor in metres, sigmaSpace in pixels (sigma_m * 100 ≈ 5 px for 5 cm).
        if bilateral_sigma > 0.0:
            sigma_px = max(1.0, bilateral_sigma * 100.0)
            fused = cv2.bilateralFilter(
                fused, d=5,
                sigmaColor=bilateral_sigma,
                sigmaSpace=sigma_px,
            )

        return fused


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accept uint8 [0,255] or float [0,1] RGB; return contiguous uint8."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)
