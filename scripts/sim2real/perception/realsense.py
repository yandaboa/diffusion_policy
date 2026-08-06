import time

import cv2
import numpy as np
import pyrealsense2 as rs


def gather_realsense_cameras(
    rgb=True,
    depth=False,
    ir=False,
    high_res_rgb=False,
    align=None,
    hardware_reset=False,
    color_wh=None,
    color_fps=30,
):
    context = rs.context()
    all_devices = list(context.devices)
    all_rs_cameras = []

    for device in all_devices:
        if hardware_reset:
            device.hardware_reset()
            time.sleep(1)
        # color_wh may be a single (w,h) applied to all, or a {serial: (w,h)} dict for
        # per-camera resolution (e.g. D435/D415 at 1080p, D455 capped at 720p).
        serial = str(device.get_info(rs.camera_info.serial_number))
        cw = color_wh.get(serial) if isinstance(color_wh, dict) else color_wh
        cf = color_fps.get(serial, 30) if isinstance(color_fps, dict) else color_fps
        rs_camera = RealSenseCamera(
            device, rgb=rgb, depth=depth, ir=ir, high_res_rgb=high_res_rgb, align=align,
            color_wh=cw, color_fps=cf
        )
        all_rs_cameras.append(rs_camera)

    return all_rs_cameras


class RealSenseCamera:
    def __init__(
        self, device, rgb=True, depth=False, ir=False, high_res_rgb=False, align=None,
        color_wh=None, color_fps=30,
    ):

        self._pipeline = rs.pipeline()
        self._serial_number = str(device.get_info(rs.camera_info.serial_number))
        self._config = rs.config()
        self._capture_clock_offset = None

        self._config.enable_device(self._serial_number)

        self.ir = ir
        self.depth = depth
        self.rgb = rgb

        # Ask librealsense to map device timestamps onto the host clock when supported.
        # The read path still has a hardware-clock fallback for older devices/firmware.
        try:
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
        except Exception:
            pass

        if self.rgb or align == "rgb":
            # explicit color_wh wins; else high_res_rgb -> 1280x720, else 640x480.
            if color_wh is not None:
                cw, ch = int(color_wh[0]), int(color_wh[1])
            elif high_res_rgb:
                cw, ch = 1280, 720
            else:
                cw, ch = 640, 480
            self._config.enable_stream(
                rs.stream.color, cw, ch, rs.format.bgr8, int(color_fps)
            )
        if self.depth or align == "depth":
            self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if self.ir:
            self._config.enable_stream(
                rs.stream.infrared, 1, 640, 480, rs.format.y8, 30
            )
            self._config.enable_stream(
                rs.stream.infrared, 2, 640, 480, rs.format.y8, 30
            )

        cfg = self._pipeline.start(self._config)
        if align == "depth":
            self._align = rs.align(rs.stream.depth)
        elif align == "color":
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        profile = self._pipeline.get_active_profile()

        self.calibration = {"intrinsics": {}}

        if self.rgb:
            color_stream = profile.get_stream(rs.stream.color)
            color_int = color_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["rgb"] = self._process_intrinsics(color_int)
        if self.depth:
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_int = depth_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["depth"] = self._process_intrinsics(
                depth_int
            )
        if self.ir:
            ir_left_stream = profile.get_stream(rs.stream.infrared, 1)
            ir_left_int = ir_left_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["ir_left"] = self._process_intrinsics(
                ir_left_int
            )
            ir_right_stream = profile.get_stream(rs.stream.infrared, 2)
            ir_right_int = ir_right_stream.as_video_stream_profile().get_intrinsics()
            self.calibration["intrinsics"]["ir_right"] = self._process_intrinsics(
                ir_right_int
            )

            # distance between the two IR cameras in meters
            extrinsics = ir_left_stream.get_extrinsics_to(ir_right_stream)
            self.calibration["ir_baseline_left_to_right"] = abs(
                extrinsics.translation[0]
            )

        self._fovy = 65

        # color_sensor = device.query_sensors()[1]
        # color_sensor.set_option(rs.option.enable_auto_exposure, True)
        # color_sensor.set_option(rs.option.exposure, 500)
        # depth_sensor = device.query_sensors()[0]

    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array(
            [[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]]
        )
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def _capture_time(self, frame, fallback):
        """Best-effort frame exposure time expressed as host epoch seconds."""
        try:
            frame_s = float(frame.get_timestamp()) * 1e-3
            domain = frame.get_frame_timestamp_domain()
            if domain == rs.timestamp_domain.system_time and abs(fallback - frame_s) < 10.0:
                return frame_s

            # Hardware-clock fallback: learn its offset from the host clock. Keeping the
            # smallest observed offset avoids baking transient USB/application delay into it.
            candidate = fallback - frame_s
            if self._capture_clock_offset is None:
                self._capture_clock_offset = candidate
            else:
                self._capture_clock_offset = min(self._capture_clock_offset, candidate)
            mapped = frame_s + self._capture_clock_offset
            return mapped if abs(fallback - mapped) < 10.0 else fallback
        except Exception:
            return fallback

    def read_camera(self):

        out = {}
        frames = self._pipeline.wait_for_frames()

        if self.ir:
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)
            out["ir_left"] = cv2.cvtColor(
                np.asanyarray(ir_left_frame.get_data()), cv2.COLOR_GRAY2RGB
            )
            out["ir_right"] = cv2.cvtColor(
                np.asanyarray(ir_right_frame.get_data()), cv2.COLOR_GRAY2RGB
            )

        if self._align is not None:
            frames = self._align.process(frames)

        if self.rgb:
            color_frame = frames.get_color_frame()
            capture_time = self._capture_time(color_frame, time.time())
            out["rgb"] = cv2.cvtColor(
                np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
            )

        if self.depth:
            depth_frame = frames.get_depth_frame()
            out["depth"] = np.asanyarray(depth_frame.get_data())

        out["read_time"] = time.time()
        out["capture_time"] = capture_time if self.rgb else out["read_time"]

        return out

    def disable_camera(self):
        self._pipeline.stop()
        self._config.disable_all_streams()
