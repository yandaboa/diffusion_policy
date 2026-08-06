"""Orbbec (Femto Bolt) camera backend mirroring ``perception/realsense.py``.

Exposes the same surface the calibration scripts rely on -- ``gather_orbbec_cameras``
plus an ``OrbbecCamera`` whose ``read_camera()`` returns ``{"rgb", "depth", "read_time"}``
and whose ``calibration["intrinsics"]["rgb"]`` holds ``cameraMatrix``/``distCoeffs`` --
so ``MultiCameraWrapper(type="orbbec")`` is a drop-in for the RealSense path.

Femto Bolt specifics (see also ``orbbec/orbbec_camera.py``):
  * No hardware depth-to-color alignment -> depth is aligned onto the color grid in
    software with ``AlignFilter(COLOR_STREAM)`` so ``depth[v, u]`` lines up with the
    color pixel at ``(u, v)`` and shares the *color* intrinsics.
  * ``read_camera()["rgb"]`` is HxWx3 uint8 RGB (the SDK already emits RGB), matching
    the RealSense backend which converts BGR->RGB.
  * ``read_camera()["depth"]`` is HxW uint16 in millimeters, so it plugs straight into
    ``depth_to_points(..., depth_scale=1000.0)`` like the RealSense z16 depth.
"""
import time

import numpy as np

from pyorbbecsdk import (
    Context, Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
    OBStreamType, AlignFilter, OBPropertyID,
)


def gather_orbbec_cameras(
    rgb=True,
    depth=False,
    ir=False,
    high_res_rgb=False,
    align=None,
):
    """Open every connected Orbbec device, one ``OrbbecCamera`` each."""
    if ir:
        raise NotImplementedError("OrbbecCamera does not expose IR streams yet")

    # Keep the Context alive for the lifetime of the cameras: the device handles it
    # hands out become invalid (OBError "NULL deviceMgr") if it is garbage-collected.
    ctx = Context()
    devices = ctx.query_devices()
    cameras = []
    for i in range(devices.get_count()):
        cam = OrbbecCamera(
            devices.get_device_by_index(i),
            rgb=rgb,
            depth=depth,
            high_res_rgb=high_res_rgb,
            align=align,
        )
        cam._context = ctx  # anchor the Context so it outlives the camera
        cameras.append(cam)
    return cameras


class OrbbecCamera:
    def __init__(self, device, rgb=True, depth=False, high_res_rgb=False, align=None):
        self.rgb = rgb
        # Depth is needed whenever the caller wants depth OR asks to align to depth/color;
        # the Femto's SW alignment requires the depth stream to be running.
        self.depth = depth or (align is not None)

        info = device.get_device_info()
        self._serial_number = str(info.get_serial_number())

        self._pipeline = Pipeline(device)
        cfg = Config()

        # Femto Bolt 4:3 RGB modes are 1280x960; pick the larger one for high_res.
        color_w, color_h, fps = (1280, 960, 30)
        color_profile = self._pick_color_profile(color_w, color_h, fps)
        cfg.enable_stream(color_profile)

        if self.depth:
            depth_profile = self._pick_depth_profile(fps)
            cfg.enable_stream(depth_profile)
            # No HW D2C on the Femto -> align depth onto the color grid in software.
            cfg.set_align_mode(OBAlignMode.DISABLE)
            self._align = AlignFilter(OBStreamType.COLOR_STREAM)
        else:
            self._align = None

        self._pipeline.start(cfg)
        self._set_color_auto(device)

        # Intrinsics live on the streamed camera params. After SW alignment depth shares
        # the color (rgb) intrinsics, so we report rgb intrinsics for both.
        cam = self._pipeline.get_camera_param()
        self.calibration = {"intrinsics": {}}
        if self.rgb:
            self.calibration["intrinsics"]["rgb"] = self._process_intrinsics(
                cam.rgb_intrinsic, cam.rgb_distortion
            )
        if self.depth:
            # depth is resampled onto the color grid -> color intrinsics describe it
            self.calibration["intrinsics"]["depth"] = self._process_intrinsics(
                cam.rgb_intrinsic, cam.rgb_distortion
            )

        # vertical FOV from the color intrinsics (RealSense backend hardcodes ~65)
        fy = cam.rgb_intrinsic.fy
        self._fovy = float(np.degrees(2.0 * np.arctan(color_profile.get_height() / (2.0 * fy))))

        self._warmup_autoexposure()

    # -------- stream-profile selection (RGB color, Y16 depth) --------
    def _pick_color_profile(self, w, h, fps):
        cpl = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        rgb = []
        for i in range(cpl.get_count()):
            sp = cpl.get_stream_profile_by_index(i).as_video_stream_profile()
            if sp.get_format() == OBFormat.RGB:
                rgb.append(sp)
        if not rgb:
            raise RuntimeError("Orbbec camera exposes no RGB-format color profile")
        for sp in rgb:
            if sp.get_width() == w and sp.get_height() == h and sp.get_fps() == fps:
                return sp
        # No exact match: stay within the requested aspect ratio if the camera offers
        # it (we need 4:3, not the camera's default 16:9), then pick the nearest area.
        target_ar = w / h
        same_ar = [s for s in rgb if abs(s.get_width() / s.get_height() - target_ar) < 0.02]
        pool = same_ar if same_ar else rgb
        pool.sort(key=lambda s: abs(s.get_width() * s.get_height() - w * h))
        chosen = pool[0]
        note = "" if same_ar else " (NO same-aspect mode; ASPECT RATIO DIFFERS)"
        print(f"[orbbec] no exact {w}x{h}@{fps} RGB profile; using "
              f"{chosen.get_width()}x{chosen.get_height()}@{chosen.get_fps()}{note}")
        return chosen

    def _pick_depth_profile(self, fps):
        dpl = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        cands = [dpl.get_stream_profile_by_index(i).as_video_stream_profile()
                 for i in range(dpl.get_count())]
        cands = [sp for sp in cands if sp.get_format() == OBFormat.Y16]
        for sp in cands:  # prefer 640x576 NFOV at requested fps
            if sp.get_width() == 640 and sp.get_height() == 576 and sp.get_fps() == fps:
                return sp
        for sp in cands:
            if sp.get_fps() == fps:
                return sp
        return cands[0]

    @staticmethod
    def _set_color_auto(device):
        """Enable color auto-exposure + auto white balance."""
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
        except Exception as e:  # not fatal; some firmwares reject these
            print(f"[orbbec] could not set auto exposure/WB: {e}")

    def _process_intrinsics(self, intr, dist):
        out = {}
        out["cameraMatrix"] = np.array(
            [[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]]
        )
        # OpenCV rational model order: [k1, k2, p1, p2, k3, k4, k5, k6]
        out["distCoeffs"] = np.array(
            [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3, dist.k4, dist.k5, dist.k6]
        )
        return out

    def _warmup_autoexposure(self, max_seconds=3.0, stable_frames=8, tol=2.0):
        """Drop frames until the color AE converges (Femto blows out the first ~1s)."""
        means, t0 = [], time.time()
        while time.time() - t0 < max_seconds:
            fs = self._pipeline.wait_for_frames(200)
            if fs is None:
                continue
            cf = fs.get_color_frame()
            if cf is None:
                continue
            means.append(float(np.frombuffer(cf.get_data(), dtype=np.uint8).mean()))
            recent = means[-stable_frames:]
            if len(recent) >= stable_frames and (max(recent) - min(recent)) < tol:
                return

    def read_camera(self):
        out = {}
        for _ in range(20):
            fs = self._pipeline.wait_for_frames(500)
            if fs is None:
                continue
            if self._align is not None:
                fs = self._align.process(fs).as_frame_set()
            cf = fs.get_color_frame() if self.rgb else None
            df = fs.get_depth_frame() if self.depth else None
            if (self.rgb and cf is None) or (self.depth and df is None):
                continue

            if self.rgb:
                out["rgb"] = np.frombuffer(cf.get_data(), dtype=np.uint8).reshape(
                    cf.get_height(), cf.get_width(), 3
                ).copy()  # already RGB
            if self.depth:
                raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(
                    df.get_height(), df.get_width()
                )
                # convert to millimeters (Femto depth_scale is ~1.0) to match z16
                depth_mm = raw.astype(np.float32) * df.get_depth_scale()
                out["depth"] = depth_mm.astype(np.uint16)
            out["read_time"] = time.time()
            return out
        raise RuntimeError("Orbbec: no complete frame within retry budget")

    def disable_camera(self):
        self._pipeline.stop()
