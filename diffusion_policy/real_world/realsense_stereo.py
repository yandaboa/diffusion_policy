"""One-shot RealSense stereo-IR (+ color) capture for the FFS depth pipeline.

FoundationStereo needs the camera's *rectified* IR stereo pair with the IR projector
DISABLED (the dot pattern corrupts learned stereo matching). The existing
``SingleRealsense`` grabs a single IR stream and aligns to color, so this is a separate,
minimal grabber used by ``ffs_depth_client`` / ``debug_pointcloud --depth-source ffs``.

Returns (left-IR, right-IR, color, K_ir, baseline_m). FFS disparity is computed in the
LEFT-IR frame, so ``K_ir`` (left infrared intrinsics) is the correct K for backprojecting
the resulting depth. The sim cam->base extrinsic is for the color optical center; left-IR
is offset from it by the small IR<->color baseline (~1-2 cm) -- ignored as part of the
sim-extrinsic approximation (see POINTCLOUD_EVAL.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StereoFrame:
    left: np.ndarray        # (H, W) uint8  left IR (rectified)
    right: np.ndarray       # (H, W) uint8  right IR (rectified)
    color: np.ndarray       # (H, W, 3) uint8 RGB, or None if want_color=False
    K_ir: np.ndarray        # (3, 3) left-IR intrinsics
    K_color: np.ndarray     # (3, 3) color intrinsics, or None
    baseline_m: float       # stereo baseline (metres)


def capture_stereo(serial: str, resolution=(1280, 720), fps: int = 30,
                   warmup: int = 30, want_color: bool = True) -> StereoFrame:
    """Grab one rectified IR stereo pair (emitter off) + optional color from a D4xx."""
    import pyrealsense2 as rs

    w, h = resolution
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)  # left
    cfg.enable_stream(rs.stream.infrared, 2, w, h, rs.format.y8, fps)  # right
    if want_color:
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipe.start(cfg)
    try:
        # Disable the IR projector so the stereo pair is clean for FoundationStereo.
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 0)

        ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intr = ir1.get_intrinsics()
        K_ir = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
        baseline = abs(ir1.get_extrinsics_to(ir2).translation[0])  # metres

        K_color = None
        if want_color:
            ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K_color = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1.0]])

        fs = None
        for _ in range(max(1, warmup)):  # let auto-exposure settle
            fs = pipe.wait_for_frames()

        left = np.asarray(fs.get_infrared_frame(1).get_data())
        right = np.asarray(fs.get_infrared_frame(2).get_data())
        color = None
        if want_color:
            color = np.asarray(fs.get_color_frame().get_data())[..., ::-1].copy()  # BGR->RGB
    finally:
        pipe.stop()

    return StereoFrame(left=left, right=right, color=color,
                       K_ir=K_ir, K_color=K_color, baseline_m=float(baseline))
