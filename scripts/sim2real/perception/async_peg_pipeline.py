"""Low-latency asynchronous camera and AprilTag pose workers.

Each camera gets two daemon threads:

* capture continuously replaces a single latest-frame slot (there is no frame queue);
* detection consumes the newest frame it has not seen, independently of other cameras.

Once a pose has been acquired, detection runs inside a pose-projected ROI. A miss triggers
an immediate full-frame reacquisition when the cooldown permits, while repeated blind scans
are rate-limited so one occluded camera cannot monopolize the CPU.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import apriltag_peg_pose as apt


def project_peg_roi(
        T_cam_peg: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        image_shape: Tuple[int, ...],
        peg_dims: Tuple[float, float, float],
        padding_px: float = 128.0,
        padding_fraction: float = 0.6,
        center_shift_px: Tuple[float, float] = (0.0, 0.0),
        expansion: float = 1.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Project the 3-D peg box and return a padded, clipped ``(x0,y0,x1,y1)`` ROI."""
    T = np.asarray(T_cam_peg, dtype=np.float64)
    dims = np.asarray(peg_dims, dtype=np.float64) / 2.0
    corners = np.array(
        [[sx * dims[0], sy * dims[1], sz * dims[2]]
         for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    cam_points = (T[:3, :3] @ corners.T).T + T[:3, 3]
    if not np.all(np.isfinite(cam_points)) or np.min(cam_points[:, 2]) <= 1e-4:
        return None

    rvec, _ = cv2.Rodrigues(T[:3, :3])
    uv, _ = cv2.projectPoints(corners, rvec, T[:3, 3], K, dist)
    uv = uv.reshape(-1, 2)
    if not np.all(np.isfinite(uv)):
        return None
    uv += np.asarray(center_shift_px, dtype=np.float64)

    lo, hi = uv.min(axis=0), uv.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = (float(padding_px) + float(padding_fraction) * float(span.max())) * float(expansion)
    lo -= pad
    hi += pad

    h, w = int(image_shape[0]), int(image_shape[1])
    x0 = max(0, int(np.floor(lo[0])))
    y0 = max(0, int(np.floor(lo[1])))
    x1 = min(w, int(np.ceil(hi[0])))
    y1 = min(h, int(np.ceil(hi[1])))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _shift_debug_corners(dbg, x0: int, y0: int):
    """Put detector debug corners from ROI coordinates back into the full image."""
    if x0 == 0 and y0 == 0:
        return dbg
    offset = np.array([x0, y0], dtype=np.float32)
    for item in dbg:
        item["corners"] = np.asarray(item["corners"]) + offset
    return dbg


class AsyncCameraPoseWorker:
    """Continuously capture and estimate one camera without blocking any other camera."""

    def __init__(
            self,
            name: str,
            camera,
            K: np.ndarray,
            dist: np.ndarray,
            T_peg_tag,
            pose_mode: str = "joint",
            use_prior: bool = True,
            prior_hold: int = 15,
            use_roi: bool = True,
            roi_padding_px: float = 128.0,
            roi_padding_fraction: float = 0.6,
            full_scan_interval_s: float = 0.25,
            aux_body=None,
            aux_interval_s: float = 0.5,
    ):
        self.name = name
        self.camera = camera
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.T_peg_tag = T_peg_tag
        self.pose_mode = pose_mode
        self.use_prior = bool(use_prior)
        self.prior_hold = int(prior_hold)
        self.use_roi = bool(use_roi)
        self.roi_padding_px = float(roi_padding_px)
        self.roi_padding_fraction = float(roi_padding_fraction)
        self.full_scan_interval_s = float(full_scan_interval_s)
        # Optional STATIC secondary body (the peg-hole). It never moves, so it is scanned
        # full-frame at a low rate on the same thread -- no ROI, no prior, negligible cost --
        # and the latest result is carried forward on every publish.
        self.aux_body = aux_body
        self.aux_interval_s = float(aux_interval_s)

        self._stop = threading.Event()
        self._frame_cond = threading.Condition()
        self._latest_frame = None
        self._frame_seq = 0
        self._result_lock = threading.Lock()
        self._latest_result = None
        self._capture_error = None
        self._detect_error = None
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name=f"peg-cap-{name}", daemon=True)
        self._detect_thread = threading.Thread(
            target=self._detect_loop, name=f"peg-detect-{name}", daemon=True)

    def start(self):
        self._capture_thread.start()
        self._detect_thread.start()
        return self

    @staticmethod
    def _frame_stamp(frame, now):
        stamp = frame.get("capture_time")
        if stamp is None:
            stamp = frame.get("read_time")
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            return now
        # Only host-epoch timestamps can be compared across cameras and with the policy process.
        return stamp if abs(now - stamp) < 60.0 else now

    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read_camera()
                if frame is None or frame.get("rgb") is None:
                    continue
                now = time.time()
                item = {
                    "rgb": frame["rgb"],
                    "capture_stamp": self._frame_stamp(frame, now),
                    "read_stamp": float(frame.get("read_time", now)),
                }
                with self._frame_cond:
                    self._frame_seq += 1
                    item["seq"] = self._frame_seq
                    self._latest_frame = item
                    self._frame_cond.notify()
                self._capture_error = None
            except Exception as exc:  # keep other cameras alive through a transient camera failure
                self._capture_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.05)

    def _wait_for_newest(self, previous_seq):
        with self._frame_cond:
            self._frame_cond.wait_for(
                lambda: self._stop.is_set()
                or (self._latest_frame is not None
                    and self._latest_frame["seq"] != previous_seq),
                timeout=0.2,
            )
            return self._latest_frame

    def _estimate(self, gray, detector, prior, roi):
        if roi is None:
            return apt.estimate_peg_pose_ex(
                gray, detector, self.K, self.dist, self.T_peg_tag,
                prior=prior, mode=self.pose_mode)

        x0, y0, x1, y1 = roi
        K_roi = self.K.copy()
        K_roi[0, 2] -= x0
        K_roi[1, 2] -= y0
        T, dbg, info = apt.estimate_peg_pose_ex(
            gray[y0:y1, x0:x1], detector, K_roi, self.dist, self.T_peg_tag,
            prior=prior, mode=self.pose_mode)
        return T, _shift_debug_corners(dbg, x0, y0), info

    @staticmethod
    def _debug_center(dbg):
        if not dbg:
            return None
        points = np.concatenate(
            [np.asarray(item["corners"], dtype=np.float64).reshape(-1, 2) for item in dbg],
            axis=0,
        )
        return 0.5 * (points.min(axis=0) + points.max(axis=0))

    def _detect_loop(self):
        while not self._stop.is_set():
            try:
                self._run_detector()
                return
            except Exception as exc:
                self._detect_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.05)

    def _run_detector(self):
        detector = apt.make_detector()  # ArucoDetector instances are not shared across threads
        aux_detector = self.aux_body.make_detector() if self.aux_body is not None else None
        last_aux_scan = -float("inf")
        aux_latest = None
        previous_seq = 0
        prior = None
        prior_age = 0
        miss_count = 0
        last_full_scan = -float("inf")
        previous_center = None
        previous_center_stamp = None
        velocity_px_s = np.zeros(2, dtype=np.float64)

        while not self._stop.is_set():
            item = self._wait_for_newest(previous_seq)
            if item is None or item["seq"] == previous_seq:
                continue
            previous_seq = item["seq"]
            rgb = item["rgb"]
            capture_stamp = item["capture_stamp"]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            started = time.perf_counter()
            now_mono = time.monotonic()

            roi = None
            if self.use_roi and prior is not None:
                prediction_dt = (
                    max(0.0, min(capture_stamp - previous_center_stamp, 0.25))
                    if previous_center_stamp is not None else 0.0
                )
                center_shift = velocity_px_s * prediction_dt
                expansion = min(4.0, 1.35 ** miss_count)
                roi = project_peg_roi(
                    prior, self.K, self.dist, gray.shape, apt.PEG_DIMS_M,
                    padding_px=self.roi_padding_px,
                    padding_fraction=self.roi_padding_fraction,
                    center_shift_px=tuple(center_shift),
                    expansion=expansion,
                )
                # Treat an almost-full image as a full scan for reacquisition throttling.
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    almost_full = ((x1 - x0) * (y1 - y0)
                                   >= 0.8 * gray.shape[0] * gray.shape[1])
                    if almost_full:
                        roi = None

            if prior is None and now_mono - last_full_scan < self.full_scan_interval_s:
                continue

            T, dbg, info = self._estimate(gray, detector, prior if self.use_prior else None, roi)
            if roi is None:
                last_full_scan = now_mono

            # An ROI miss gets a full-resolution, full-frame chance. Repeated blind scans are
            # rate-limited; capture continues replacing its one-slot buffer while this runs.
            fallback_full = False
            if T is None and roi is not None and now_mono - last_full_scan >= self.full_scan_interval_s:
                fallback_full = True
                last_full_scan = now_mono
                T, dbg, info = self._estimate(
                    gray, detector, prior if self.use_prior else None, None)

            if T is not None:
                prior = T
                prior_age = 0
                miss_count = 0
                center = self._debug_center(dbg)
                if center is not None and previous_center is not None:
                    center_dt = capture_stamp - previous_center_stamp
                    if 1e-3 < center_dt < 1.0:
                        measured = (center - previous_center) / center_dt
                        measured = np.clip(measured, -5000.0, 5000.0)
                        velocity_px_s = 0.5 * velocity_px_s + 0.5 * measured
                if center is not None:
                    previous_center = center
                    previous_center_stamp = capture_stamp
            else:
                miss_count += 1
                prior_age += 1
                if prior_age > self.prior_hold:
                    prior = None
                    velocity_px_s[:] = 0.0

            # Static secondary body (peg-hole): low-rate full-frame scan. Its cost is amortized
            # over aux_interval_s, so it cannot slow the peg's per-frame tracking.
            if aux_detector is not None and now_mono - last_aux_scan >= self.aux_interval_s:
                last_aux_scan = now_mono
                T_aux, dbg_aux, info_aux = self.aux_body.estimate(
                    gray, aux_detector, self.K, self.dist)
                if T_aux is not None:
                    aux_latest = {
                        "T_cam_aux": T_aux,
                        "tags": sorted(int(d["id"]) for d in dbg_aux),
                        "dbg": dbg_aux,
                        "conf": float(info_aux["conf"]),
                        "n_tags": int(info_aux["n_tags"]),
                        "reproj_rms": info_aux["reproj_rms"],
                        "capture_stamp": capture_stamp,
                    }

            done_stamp = time.time()
            result = {
                "name": self.name,
                "frame_seq": item["seq"],
                "capture_stamp": capture_stamp,
                "read_stamp": item["read_stamp"],
                "done_stamp": done_stamp,
                "T_cam_peg": T,
                "dbg": dbg,
                "info": info,
                "tags": sorted(int(d["id"]) for d in dbg),
                "detect_ms": (time.perf_counter() - started) * 1000.0,
                "capture_to_done_ms": (done_stamp - capture_stamp) * 1000.0,
                "roi": roi,
                "used_roi": roi is not None and not fallback_full,
                "fallback_full": fallback_full,
                "aux": aux_latest,          # latest static-body estimate, carried forward
            }
            with self._result_lock:
                self._latest_result = result
            self._detect_error = None

    def latest_result(self) -> Optional[Dict[str, Any]]:
        with self._result_lock:
            return None if self._latest_result is None else dict(self._latest_result)

    def latest_frame(self) -> Optional[Dict[str, Any]]:
        with self._frame_cond:
            return None if self._latest_frame is None else dict(self._latest_frame)

    @property
    def error(self):
        return self._detect_error or self._capture_error

    def stop(self, timeout=1.0):
        self._stop.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        self._detect_thread.join(timeout=timeout)
        self._capture_thread.join(timeout=timeout)
