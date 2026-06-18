"""
Core Orbbec Femto Bolt camera API: open the stream, software-align depth to
color, warm up auto-exposure, grab synchronized frames, and turn a frame set
into a dense point cloud / intrinsics. No segmentation, no post-processing --
just the pieces for *talking to the camera*.

Runs anywhere the `pyorbbecsdk2` wheel is installed (imports as `pyorbbecsdk`);
the only other dependency is numpy. Higher-level scripts (e.g.
`orbbec_segment_pointclouds.py`) import these helpers and add their own
perception / post-processing on top.

Femto Bolt specifics this module hides:
  * No hardware depth-to-color alignment -> we align in software with
    `AlignFilter(COLOR_STREAM)`, so depth is resampled onto the color grid and
    every (u,v) color pixel has a depth at depth[v,u].
  * `PointCloudFilter` then emits a DENSE (H*W, 6) array (x,y,z,r,g,b),
    row-major over that same color grid -- reshape to (H,W,6) and an image-space
    mask indexes the cloud directly.
  * Positions come out in millimeters (caller converts to meters as needed).

Typical use:
    pipe, align = open_camera(serial=None, color_w=1280, color_h=720, fps=30)
    try:
        warmup_autoexposure(pipe, align)
        color, fs = capture_aligned(pipe, align)        # HxWx3 uint8, frameset
        K, cam = color_intrinsics(pipe)                 # 3x3, OBCameraParam
        pcf = make_pointcloud_filter(cam)
        grid = orbbec_pointcloud(pcf, fs)               # (H,W,6) xyz(mm)+rgb
    finally:
        pipe.stop()
"""
import time

import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, OBStreamType,
    AlignFilter, PointCloudFilter, OBPropertyID,
)


# ----------------------------------------------------------------------------
# stream-profile selection
# ----------------------------------------------------------------------------
def _pick_color_profile(pipe, w, h, fps):
    """Find an RGB-format color profile at (w,h,fps); fall back to nearest RGB."""
    cpl = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    rgb = []
    for i in range(cpl.get_count()):
        sp = cpl.get_stream_profile_by_index(i).as_video_stream_profile()
        if sp.get_format() == OBFormat.RGB:
            rgb.append(sp)
    if not rgb:
        raise RuntimeError('camera exposes no RGB-format color profile')
    for sp in rgb:
        if sp.get_width() == w and sp.get_height() == h and sp.get_fps() == fps:
            return sp
    rgb.sort(key=lambda s: abs(s.get_width() * s.get_height() - w * h))
    chosen = rgb[0]
    print(f'[orbbec] no exact {w}x{h}@{fps} RGB profile; using '
          f'{chosen.get_width()}x{chosen.get_height()}@{chosen.get_fps()}')
    return chosen


def _pick_depth_profile(pipe, fps):
    """Pick a Y16 depth profile, preferring the requested fps (640x576 NFOV)."""
    dpl = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    cands = []
    for i in range(dpl.get_count()):
        sp = dpl.get_stream_profile_by_index(i).as_video_stream_profile()
        if sp.get_format() == OBFormat.Y16:
            cands.append(sp)
    # prefer the 640x576 mode at the requested fps, else any matching fps, else first
    for sp in cands:
        if sp.get_width() == 640 and sp.get_height() == 576 and sp.get_fps() == fps:
            return sp
    for sp in cands:
        if sp.get_fps() == fps:
            return sp
    return cands[0]


def _set_color_exposure(dev, exposure):
    """exposure is None -> auto-exposure+auto-WB on; else fix exposure to that value."""
    if exposure is None:
        dev.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
        dev.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
    else:
        dev.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        dev.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, int(exposure))


# ----------------------------------------------------------------------------
# open / warm up / capture
# ----------------------------------------------------------------------------
def open_camera(serial, color_w, color_h, fps, exposure=None):
    """Start color+depth, software-aligned to the color frame. Returns
    (pipeline, align_filter)."""
    pipe = Pipeline()
    cfg = Config()
    color = _pick_color_profile(pipe, color_w, color_h, fps)
    depth = _pick_depth_profile(pipe, fps)
    cfg.enable_stream(color)
    cfg.enable_stream(depth)
    # Femto Bolt has no HW D2C -> align depth onto the color grid in software.
    cfg.set_align_mode(OBAlignMode.DISABLE)
    align_filter = AlignFilter(OBStreamType.COLOR_STREAM)
    pipe.start(cfg)
    _set_color_exposure(pipe.get_device(), exposure)
    print(f'[orbbec] streaming color {color.get_width()}x{color.get_height()} '
          f'+ depth {depth.get_width()}x{depth.get_height()} (SW-aligned to color)')
    return pipe, align_filter


def warmup_autoexposure(pipe, align_filter, max_seconds=3.0, stable_frames=8,
                        tol=2.0):
    """Discard frames until the color auto-exposure converges. The Femto Bolt's
    color AE blows the frame out for ~0.5s after the stream starts and only
    settles after ~1s (~30 frames) -- grabbing too early gives a blinding-white
    image. We watch the mean brightness and return once it stops changing (spread
    < tol over `stable_frames` consecutive frames), capped at max_seconds."""
    means = []
    t0 = time.time()
    n = 0
    while time.time() - t0 < max_seconds:
        fs = pipe.wait_for_frames(200)
        if fs is None:
            continue
        cf = fs.get_color_frame()
        if cf is None:
            continue
        img = np.frombuffer(cf.get_data(), dtype=np.uint8)
        means.append(float(img.mean()))
        n += 1
        recent = means[-stable_frames:]
        if len(recent) >= stable_frames and (max(recent) - min(recent)) < tol:
            print(f'[orbbec] auto-exposure settled after {n} frames '
                  f'({time.time() - t0:.1f}s), mean brightness {recent[-1]:.0f}')
            return
    print(f'[orbbec] warmup hit {max_seconds:.1f}s cap (mean brightness '
          f'{means[-1] if means else float("nan"):.0f}); proceeding')


def capture_aligned(pipe, align_filter, timeout_ms=500, tries=20):
    """Grab one synchronized frame set, software-align depth->color, and return
    (color_rgb HxWx3 uint8, aligned_frameset). Retries until both frames land."""
    for _ in range(tries):
        fs = pipe.wait_for_frames(timeout_ms)
        if fs is None:
            continue
        fs = align_filter.process(fs).as_frame_set()
        cf = fs.get_color_frame()
        df = fs.get_depth_frame()
        if cf is None or df is None:
            continue
        color = np.frombuffer(cf.get_data(), dtype=np.uint8).reshape(
            cf.get_height(), cf.get_width(), 3)
        return color.copy(), fs
    raise RuntimeError('no aligned color+depth frame within retry budget')


# ----------------------------------------------------------------------------
# intrinsics / point cloud
# ----------------------------------------------------------------------------
def color_intrinsics(pipe):
    """Return (K 3x3 float32, OBCameraParam) for the color stream. K maps a
    point in the color-camera frame to its (u,v) pixel."""
    cam = pipe.get_camera_param()
    K = np.array([[cam.rgb_intrinsic.fx, 0, cam.rgb_intrinsic.cx],
                  [0, cam.rgb_intrinsic.fy, cam.rgb_intrinsic.cy],
                  [0, 0, 1]], np.float32)
    return K, cam


def make_pointcloud_filter(cam):
    """Build a PointCloudFilter wired to this camera, emitting RGB points."""
    pcf = PointCloudFilter()
    pcf.set_camera_param(cam)
    pcf.set_create_point_format(OBFormat.RGB_POINT)
    return pcf


def orbbec_pointcloud(pcf, frameset):
    """Run Orbbec's PointCloudFilter on an aligned frame set. Returns the dense
    (H, W, 6) grid [x,y,z (mm), r,g,b (0-255)] row-major over the color frame."""
    df = frameset.get_depth_frame()
    cf = frameset.get_color_frame()
    pcf.set_position_data_scaled(df.get_depth_scale())   # positions in mm
    pts_frame = pcf.process(frameset)
    arr = np.array(pcf.calculate(pts_frame), dtype=np.float32)   # (H*W, 6)
    return arr.reshape(cf.get_height(), cf.get_width(), 6)
