from typing import Optional
import contextlib
import pathlib
import numpy as np
import time
import shutil
import math
import threading
from collections import deque
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from diffusion_policy.real_world.ur5e_kinematics import (
    axis_angle_to_quat, get_ee_pose, quat_to_axis_angle)

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualQ': 'arm_joint_pos',
    'ActualTCPPose': 'end_effector_pose',  # EE pose [x,y,z,rx,ry,rz] from robot
    'ActualTCPForce': 'tcp_force',  # 6D wrench [Fx,Fy,Fz,Tx,Ty,Tz] from F/T sensor
    'gripper_current': 'gripper_current',  # motor current (0-255, ~10mA/unit), proxy for grip force
    'gripper_pos': 'gripper_pos',          # raw POS register (0=open, 255=closed)
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}


@contextlib.contextmanager
def _open_front_realsense_depth(serial, resolution):
    """Persistent front-D455 hardware-depth stream for the point-cloud obs path.

    Twin of ``debug_pointcloud._open_realsense_stream``: keeps one ``SingleRealsense`` open and
    yields ``grab() -> (color RGB, depth, K, units/m)``. Used only when ``setup_pointcloud`` is
    called with ``depth_source='realsense'`` (the FFS path uses ``open_stereo_stream`` instead).
    Owns its own SharedMemoryManager so it never collides with the env's MultiRealsense.
    """
    shm = SharedMemoryManager(); shm.start()
    cam = SingleRealsense(shm, serial, resolution=tuple(resolution),
                          enable_color=True, enable_depth=True)
    cam.start(wait=True); cam.start_wait()
    try:
        for _ in range(30):  # let auto-exposure settle before the first grab
            cam.get()
        K = cam.get_intrinsics()
        units_per_m = 1.0 / cam.get_depth_scale()

        def grab():
            out = cam.get()
            color = out["color"][..., ::-1].copy()  # BGR->RGB
            return color, out["depth"], K, units_per_m

        yield grab
    finally:
        cam.stop(wait=True); shm.shutdown()


@contextlib.contextmanager
def _open_front_orbbec(serial, resolution, flying_pixel_thresh_mm=None):
    """Persistent front-Orbbec (Femto Bolt) stream for the point-cloud obs path.

    Orbbec twin of ``_open_front_realsense_depth``. Opens color + depth, software-aligns depth
    onto the color grid (no HW D2C on the Femto), and yields
    ``grab() -> (color RGB HxWx3, xyz_m HxWx3, rgb HxWx3, K_color)``.

    The points come straight from Orbbec's ``PointCloudFilter`` -- a dense grid row-major over the
    color frame, already in the COLOR optical frame -- so they line up 1:1 with the SAM2 masks
    (which run on that same color image) and need NO IR->color warp. Positions are converted mm->m.
    Only one Orbbec is supported (``open_camera`` opens the first device); ``serial`` is advisory.

    ``flying_pixel_thresh_mm``: if set, ToF flying pixels (mixed pixels streaking between fg
    edges and the background) are removed from the grid by local depth-range thresholding
    (``pointcloud_builder.flying_pixel_mask``); their xyz is zeroed, i.e. the same "invalid"
    convention (z=0) the cloud build already discards. None leaves the grid unfiltered.
    """
    from orbbec.orbbec_camera import (
        open_camera, warmup_autoexposure, capture_aligned, color_intrinsics,
        make_pointcloud_filter, orbbec_pointcloud)
    from diffusion_policy.real_world.pointcloud_builder import flying_pixel_mask
    w, h = tuple(resolution)
    pipe, align = open_camera(serial, w, h, 30)
    try:
        warmup_autoexposure(pipe, align)
        K, cam = color_intrinsics(pipe)
        pcf = make_pointcloud_filter(cam)

        def grab():
            color, fs = capture_aligned(pipe, align)
            grid = orbbec_pointcloud(pcf, fs)            # (H,W,6) xyz(mm) + rgb
            xyz_m = (grid[..., :3] / 1000.0).astype(np.float32)   # mm -> m, color frame
            if flying_pixel_thresh_mm is not None:
                flying = flying_pixel_mask(
                    xyz_m[..., 2], thresh=flying_pixel_thresh_mm * 1e-3)
                xyz_m[flying] = 0.0
            rgb = grid[..., 3:6].astype(np.uint8)
            return color, xyz_m, rgb, K

        yield grab
    finally:
        pipe.stop()


def _rotmat_to_quat_wxyz(R):
    """(3,3) rotation matrix -> (w,x,y,z) unit quaternion (numpy, no scipy dep)."""
    m = np.asarray(R, np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], np.float64)
    return (q / np.linalg.norm(q)).astype(np.float32)


class RealEnv:
    def __init__(self,
            # required params
            output_dir,
            robot_ip,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            camera_configs=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            rolling_action_buffer=False,
            action_mode='joint',  # 'joint' or 'cartesian' (direct Cartesian OSC)
            # width of the *arm* part of the policy action (excludes the 1 gripper dim).
            # 6 = full 6-DOF Cartesian delta [x,y,z,rx,ry,rz]; 3 = position-only [x,y,z].
            # Drives the shape of last_arm_action obs and the obs_actions buffer.
            arm_action_dim=6,
            # robot
            init_joints=True,
            custom_init_joints=None,  # Custom initial joint positions
            # OSC parameters
            osc_kp_pos=1000.0,
            osc_kp_rot=50.0,
            osc_damping_ratio_pos=1.0,
            osc_damping_ratio_rot=1.0,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(640,480),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(640,480),
            # depth
            enable_depth=False,
            # shared memory
            shm_manager=None,
            rescale_pixels=True,
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            if rescale_pixels:
                color_transform = lambda x: color_tf(x).astype(np.float32) / 255
            else:
                color_transform = lambda x: color_tf(x).astype(np.float32)

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
        
        if len(camera_serial_numbers) > 0:
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(camera_serial_numbers),
                in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
                max_resolution=multi_cam_vis_resolution
            )
        else:
            # Camera-less env (state policy: peg pose comes from the external
            # peg_fusion_viz --publish worker). Nothing to tile.
            rw, rh, col, row = obs_image_resolution[0], obs_image_resolution[1], 1, 1
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )
        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            advanced_mode_config=camera_configs,
            enable_color=True,
            enable_depth=enable_depth,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            )
        
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])

        # Fallback home pose (radians) used only when the caller doesn't pass custom_init_joints.
        default_init_joints = np.array([0.020015, -1.39426, 2.13054, -2.246, -1.618852138519287, -0.087])

        # Handle joint initialization
        j_init = None
        if init_joints:
            if custom_init_joints is not None:
                # Use caller-provided joint positions (e.g. the sim joint pose).
                j_init = np.array(custom_init_joints)
                print(f"Using custom initial joint positions: {j_init}")
            else:
                # Fall back to the hardcoded default home pose.
                j_init = default_init_joints
                print(f"Using default initial joint positions: {j_init}")

        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=500,
            launch_timeout=3,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
            # OSC parameters
            osc_kp_pos=osc_kp_pos,
            osc_kp_rot=osc_kp_rot,
            osc_damping_ratio_pos=osc_damping_ratio_pos,
            osc_damping_ratio_rot=osc_damping_ratio_rot,
            )

        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map
        self.action_mode = action_mode
        self.arm_action_dim = arm_action_dim
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # state-policy peg-pose reader (set by setup_state_pose_reader); reads the pose
        # published by the decoupled peg_fusion_viz --publish worker.
        self.state_pose_reader = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        # No-timestamp action buffer
        rolling_action_buffer = self.action_buffer = deque(maxlen=self.n_obs_steps) if rolling_action_buffer else None
        # Canonical last raw action (arm_action_dim + 1 gripper) the policy issued, updated on
        # every exec_actions call so prev_action never stalls. Single source of truth for the
        # eval scripts' prev_action obs; reset to zeros each episode (matches sim last_action).
        self.last_action = None

        self.start_time = None

        # Point-cloud obs pipeline (front D455 stereo/FFS + SAM2 streaming). Set up lazily by
        # ``setup_pointcloud`` and torn down in ``stop``; ``None`` until then so plain RGB/depth
        # eval is unaffected. See ``get_obs_pc`` / debug_pointcloud.py for the same pipeline.
        self._pc_stack = None
        self._pc = None

        # FoundationPose object-pose pipeline (front camera -> FoundationPose worker in its own
        # conda env). Set up lazily by ``setup_pose_estimation`` and torn down in ``stop``; ``None``
        # until then. Independent of the point-cloud pipeline so pose can be tested standalone.
        self._pose_stack = None
        self._pose = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        try:
            self.end_episode()
        except Exception as e:
            # Teardown must always reach the camera/robot shutdown below, otherwise the
            # RTDE socket and the realsense subprocesses are left dangling.
            print(f'[RealEnv] end_episode failed during stop: {e}')
        if self._pc_stack is not None:
            self._pc_stack.close()  # close front stereo/FFS stream + SAM2 worker
            self._pc_stack = None
            self._pc = None
        if self._pose_stack is not None:
            if self._pose is not None:
                self._pose['stop_event'].set()  # ask the background tracking thread to exit
                th = self._pose.get('thread')
                if th is not None:
                    th.join(timeout=5)
                try:
                    self._pose['client'].stop()  # kill FoundationPose worker subprocess + free shm
                except Exception:
                    pass
            self._pose_stack.close()  # close the front-camera stream
            self._pose_stack = None
            self._pose = None
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, modality: str = 'rgb') -> dict:
        """Shared observation entry point; dispatches by ``modality``.

        - 'rgb'   : camera RGB (front/side/wrist) + robot state + last actions (default;
                    identical to the historical ``get_obs()`` used by the image/tactile evals).
        - 'state' : robot state + last actions + latest fused peg pose (``T_base_peg``) read
                    from the decoupled peg_fusion_viz --publish worker. Requires
                    ``setup_state_pose_reader(...)`` first.
        - 'pc'    : point-cloud obs (see ``get_obs_pc``; requires ``setup_pointcloud``).
        - 'pose'  : 6-DoF object pose obs (see ``get_obs_pose``; requires ``setup_pose_estimation``).

        Each modality reuses the same raw camera/robot grab + timestamp alignment and the
        same robot-state assembly; only the perception branch differs.
        """
        if modality == 'rgb':
            return self.get_obs_rgb()
        if modality == 'state':
            return self.get_obs_state()
        if modality == 'pc':
            return self.get_obs_pc()
        if modality == 'pose':
            return self.get_obs_pose()
        raise ValueError(f"unknown obs modality {modality!r}; "
                         "expected one of 'rgb', 'state', 'pc', 'pose'.")

    def _grab_and_align(self):
        """Grab the latest camera + robot data and compute the aligned obs timestamps.

        Returns ``obs_align_timestamps`` (n_obs_steps,) and ``last_robot_data``; the camera
        data is stashed on ``self.last_realsense_data``. Shared by every ``get_obs_*`` path.
        """
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)
        last_robot_data = self.robot.get_all_state()
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        return obs_align_timestamps, last_robot_data

    @staticmethod
    def _nearest_idxs(this_timestamps, obs_align_timestamps):
        """Per aligned timestamp, the index of the newest sample strictly before it."""
        this_idxs = []
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idxs.append(is_before_idxs[-1] if len(is_before_idxs) > 0 else 0)
        return this_idxs

    def _assemble_robot_obs(self, last_robot_data, obs_align_timestamps):
        """Robot proprio (remapped via ``obs_key_map``) + last_arm/gripper action buffer.

        Shared by the 'rgb' and 'state' modalities. Also feeds the obs accumulator.
        """
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_idxs = self._nearest_idxs(robot_timestamps, obs_align_timestamps)

        robot_obs_raw = dict()
        for key, v in last_robot_data.items():
            if key in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[key]] = v
        robot_obs = {key: v[this_idxs] for key, v in robot_obs_raw.items()}

        if self.obs_accumulator is not None:
            self.obs_accumulator.put(robot_obs_raw, robot_timestamps)

        # last_arm_action / last_gripper_action from the action buffer. When exec_actions is
        # called with obs_actions=<raw policy output>, these are the RAW (pre-scale) actions.
        # Buffer rows are [arm_action_dim arm dims, 1 gripper dim]; width adapts to the policy.
        arm_dim = self.arm_action_dim
        act_dim = arm_dim + 1
        if self.action_buffer is not None and len(self.action_buffer) > 0:
            last_actions_raw = np.zeros((self.n_obs_steps, act_dim), dtype=np.float32)
            actions = np.array(self.action_buffer)
            last_actions_raw[:actions.shape[0], :] = actions
            last_actions = {
                'last_arm_action': last_actions_raw[:, :arm_dim],
                'last_gripper_action': last_actions_raw[:, arm_dim:act_dim],
            }
        else:
            obs_array = np.zeros((self.n_obs_steps, act_dim), dtype=np.float32)
            last_actions = {
                'last_arm_action': obs_array[:, :arm_dim],
                'last_gripper_action': np.zeros((self.n_obs_steps, 1)),
            }
        return robot_obs, last_actions

    def get_obs_rgb(self) -> dict:
        "Camera RGB + robot state + last actions (the historical default obs)."
        assert self.is_ready
        obs_align_timestamps, last_robot_data = self._grab_and_align()

        camera_obs = dict()
        n_cams = self.realsense.n_cameras
        cam_names = ['front_rgb', 'side_rgb', 'wrist_rgb'] if n_cams >= 3 \
            else ['side_rgb', 'wrist_rgb']
        for camera_idx, value in self.last_realsense_data.items():
            this_idxs = self._nearest_idxs(value['timestamp'], obs_align_timestamps)
            if camera_idx < len(cam_names):
                camera_obs[cam_names[camera_idx]] = value['color'][this_idxs]

        robot_obs, last_actions = self._assemble_robot_obs(last_robot_data, obs_align_timestamps)

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(last_actions)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def get_obs_state(self) -> dict:
        """Robot state + last actions + the latest fused peg pose for the low-dim state policy.

        The peg pose is produced by the DECOUPLED, tried-and-tested fusion worker
        ``scripts/sim2real/perception/peg_fusion_viz.py --publish --headless`` (it owns the
        cameras at their calibrated resolutions and runs the full fusion at its own rate); this
        method just fetches the latest published pose (non-blocking) via
        ``setup_state_pose_reader(...)``. Runs camera-less -- the env holds no cameras. Adds:
            peg_pose_base:   (4,4) float64 fused peg->base, or None if no fresh detection
            peg_seen:        bool (a fresh, non-stale publish is available)
            peg_ncams:       int cameras that contributed to the published fuse
            peg_stamp_age_s: seconds since the worker's last publish (inf if missing)
            peg_ntag_detections: visible tag detections summed across cameras
            peg_tag_ids:     sorted unique visible tag IDs
        """
        assert self.is_ready
        if self.state_pose_reader is None:
            raise RuntimeError("get_obs_state requires setup_state_pose_reader(...) first.")
        # Camera-less: align obs timestamps to the robot receive stream (no camera stream).
        last_robot_data = self.robot.get_all_state()
        dt = 1 / self.frequency
        last_ts = float(np.asarray(last_robot_data['robot_receive_timestamp']).reshape(-1)[-1])
        obs_align_timestamps = last_ts - (np.arange(self.n_obs_steps)[::-1] * dt)
        robot_obs, last_actions = self._assemble_robot_obs(last_robot_data, obs_align_timestamps)

        peg = self.state_pose_reader.read()
        obs_data = dict(robot_obs)
        obs_data.update(last_actions)
        obs_data['peg_pose_base'] = peg['T_base_peg']
        obs_data['peg_seen'] = peg['seen']
        obs_data['peg_ncams'] = peg['ncams']
        obs_data['peg_stamp_age_s'] = peg['age_s']
        obs_data['peg_capture_stamp'] = peg['capture_stamp']
        obs_data['peg_capture_age_s'] = peg['capture_age_s']
        obs_data['peg_ntag_detections'] = peg['ntag_detections']
        obs_data['peg_tag_ids'] = peg['tag_ids']
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def setup_state_pose_reader(self, state_file, stale_after_s=0.5):
        """Point ``get_obs_state`` at the peg pose published by the decoupled fusion worker.

        Run the worker separately (owns the cameras, tried-and-tested logic/resolutions):
            python scripts/sim2real/perception/peg_fusion_viz.py --publish [PATH] --headless
        which atomically writes ``{stamp, joints, T_base_peg, ...}`` to ``state_file`` each loop.
        Pose estimation therefore runs at the worker's own (higher) rate, fully decoupled from
        this control loop, which only fetches the latest value.
        """
        from diffusion_policy.real_world.peg_pose_reader import PublishedPegPose
        self.state_pose_reader = PublishedPegPose(state_file, stale_after_s=stale_after_s)
        print(f"[state] reading fused peg pose from {state_file} (stale_after={stale_after_s}s)")
        return self.state_pose_reader

    # ========= point-cloud obs API ===========
    def setup_pointcloud(self,
            front_serial,
            extrinsic,
            depth_source='orbbec',
            resolution=(1280, 720),
            sam2_ckpt='orbbec/weights/sam2/sam2.1_hiera_base_plus.pt',
            sam2_cfg='configs/sam2.1/sam2.1_hiera_b+.yaml',
            erode=3,
            crop_lo=None, crop_hi=None,
            budget=None,
            prompts=None,
            prompt_classes=None,
            ffs_mock=False,
            segment=True,
            flying_pixel_mm=None,
            device=None):
        """Stand up the front-camera point-cloud pipeline used by ``get_obs_pc``.

        Mirrors ``debug_pointcloud.py --video``: opens a persistent front-D455 stream
        (stereo->FFS metric depth, or hardware depth), starts the SAM2 streaming segmenter, and
        seeds it ONCE on the first frame (interactive click-prompts unless ``prompts`` is given).
        Thereafter ``get_obs_pc`` tracks + rebuilds the EE-frame segmented cloud per call.

        The front camera is owned by THIS pipeline, not the env's MultiRealsense -- keep the
        front serial out of ``camera_serial_numbers`` so the device isn't opened twice.

        Args:
            front_serial: serial of the front D455 (stereo IR + color).
            extrinsic: (4,4) camera->base array, or a path to a .npy.
            depth_source: 'ffs' (stereo + Fast-FoundationStereo) or 'realsense' (hardware depth).
            resolution: capture (w, h).
            sam2_ckpt/sam2_cfg: SAM2 streaming-fork weights/config.
            erode: per-class mask erosion (px) before stamping the label map.
            crop_lo/crop_hi: optional EE-frame AABB (metres) applied in build_cloud.
            budget: per-class point budget (defaults to pointcloud_builder.DEFAULT_BUDGET).
            prompts: {class -> (pos, neg)} clicks; None opens one matplotlib window per class.
            prompt_classes: which classes to interactively prompt when ``prompts`` is None (one
                SAM2-tracked window each). Defaults to robot/peg/hole; pass e.g. ('peg', 'hole')
                to skip the robot (peg/hole-only policies). Ignored when ``prompts`` is given.
            ffs_mock: ramp depth for plumbing tests without FFS weights.
            segment: run SAM2 (interactive prompts + streaming tracker). True for the policy obs
                path. Set False to skip segmentation entirely -- no SAM2 load, no click prompts --
                so ``grab_segmented_cloud_camera_frame`` returns the FULL raw cloud (every valid
                point, label NaN). Used by the perception-gap probe. ``get_obs_pc`` requires True.
            flying_pixel_mm: Orbbec only -- drop flying pixels whose 3x3 local depth range
                exceeds this many mm (see ``_open_front_orbbec``). None disables. Ignored for
                'ffs'/'realsense' depth sources.
            device: torch device for the on-GPU cloud build (defaults to the SAM2 device, or
                cuda/cpu autodetect when ``segment`` is False).
        """
        from diffusion_policy.real_world.pointcloud_segmenter import (
            StreamingSegmenter, pick_prompts_interactive)

        T_cam_base = np.load(extrinsic) if isinstance(extrinsic, str) else np.asarray(extrinsic)
        stack = contextlib.ExitStack()

        # ``direct_points``: the depth source emits camera-frame XYZ directly (Orbbec
        # PointCloudFilter) instead of a depth image we backproject. ``collect_depth`` then
        # returns ``(xyz_m_grid, rgb_grid)`` rather than a depth array. See get_obs_pc.
        direct_points = False

        if depth_source == 'ffs':
            from diffusion_policy.real_world.realsense_stereo import open_stereo_stream
            from diffusion_policy.real_world.ffs_depth_client import FFSDepthClient
            grab_stereo = stack.enter_context(
                open_stereo_stream(front_serial, tuple(resolution), want_color=True))
            ffs = stack.enter_context(FFSDepthClient(mock=ffs_mock))

            def grab():
                f = grab_stereo()
                ffs.submit(f.left, f.right, f.K_ir, f.baseline_m)  # depth runs while we track
                warp = {"K_ir": f.K_ir, "K_color": f.K_color, "T_ir_color": f.T_ir_color}
                return f.color, f.K_ir, 1.0, warp  # depth already metric -> units/m = 1

            def collect_depth():
                return ffs.collect()
        elif depth_source == 'realsense':
            grab_rs = stack.enter_context(_open_front_realsense_depth(front_serial, resolution))
            pending = {}

            def grab():
                color, depth, K, ups = grab_rs()  # hardware depth: synchronous
                pending['depth'] = depth
                return color, K, ups, None

            def collect_depth():
                return pending.pop('depth')
        elif depth_source == 'orbbec':
            if flying_pixel_mm is not None:
                print(f"[RealEnv] Orbbec flying-pixel filter ON "
                      f"(3x3 local range > {flying_pixel_mm:.0f} mm -> dropped)")
            grab_ob = stack.enter_context(_open_front_orbbec(
                front_serial, resolution, flying_pixel_thresh_mm=flying_pixel_mm))
            pending = {}

            def grab():
                color, xyz_m, rgb, K = grab_ob()
                pending['xyz'] = xyz_m
                pending['rgb'] = rgb
                # color frame -> warp None (masks compose directly); depth_scale unused for
                # direct points (XYZ already metric), but kept in the tuple for interface parity.
                return color, K, 1.0, None

            def collect_depth():
                return pending.pop('xyz'), pending.pop('rgb')  # (H,W,3) m, (H,W,3) rgb
            direct_points = True
        else:
            stack.close()
            raise ValueError(
                f"depth_source must be 'ffs', 'realsense' or 'orbbec', got {depth_source!r}")

        try:
            color, K, depth_scale, warp = grab()
            if segment:
                if prompts is None:
                    prompts = pick_prompts_interactive(color) if prompt_classes is None \
                        else pick_prompts_interactive(color, class_names=tuple(prompt_classes))
                if not prompts:
                    raise RuntimeError("setup_pointcloud: no SAM2 prompts (need >=1 class clicked)")
                seg = StreamingSegmenter(sam2_ckpt, sam2_cfg, device=device)
                seg.start(color, prompts)  # seed tracker on frame 0
                dev = seg.device
            else:
                # Full-cloud / perception-gap mode: no SAM2, no prompts.
                seg = None
                import torch
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            _ = collect_depth()  # reap (+ discard) the frame-0 depth submitted by grab()
        except Exception:
            stack.close()
            raise

        self._pc_stack = stack
        self._pc = dict(
            grab=grab, collect_depth=collect_depth, seg=seg, device=dev,
            T_cam_base=T_cam_base, erode=erode, crop_lo=crop_lo, crop_hi=crop_hi,
            budget=budget, depth_source=depth_source, direct_points=direct_points,
        )
        print(f"[RealEnv] point-cloud pipeline up: depth_source={depth_source}, "
              f"segment={segment}, classes={list(prompts) if segment else '[]'}, device={dev}")

    def get_obs_pc(self) -> dict:
        """One segmented EE-frame point cloud + the robot state the BC PointNet needs.

        Runs the same per-frame pipeline as ``debug_pointcloud.py --video`` (grab -> SAM2 track
        -> warp/compose label map -> ``build_cloud_torch``), using the CURRENT EE pose from FK so
        the cloud frame tracks the moving arm. ``PointNetPolicy`` is single-cloud (no n_obs
        history), so this returns one cloud per call.

        Returns a dict (parse it for ``policy.predict_from_state``):
            point_cloud:        (num_points, 4) float32, EE frame, 4th channel = seg label.
            arm_joint_pos:      (6,) float32 current arm joints (rad).
            gripper_pos:        float32 normalized gripper position (0=open..1=closed).
            end_effector_pose:  (6,) float32 [xyz, axis-angle], wrist_3_link in base frame.
            cloud_stats:        CloudStats (per-class realized/available, AABB) for logging.
            color, masks:       front color frame + per-class mask tensors (optional vis/video).
            timestamp:          float wall-clock at grab.
        """
        assert self._pc is not None, "call setup_pointcloud() before get_obs_pc()"
        assert self._pc['seg'] is not None, \
            "get_obs_pc needs the SAM2 segmenter; call setup_pointcloud(segment=True)"
        import torch
        from diffusion_policy.real_world import pointcloud_builder as B
        from diffusion_policy.real_world.pointcloud_segmenter import (
            compose_label_map_torch, warp_masks_to_depth_frame_torch)
        pc = self._pc

        # current robot state -> EE pose via calibrated FK (cloud frame must track the arm)
        state = self.robot.get_state()
        arm_joint_pos = np.asarray(state['ActualQ'], np.float64)[:6]
        gripper_pos_raw = float(np.asarray(state['gripper_pos']).reshape(-1)[0]) \
            if 'gripper_pos' in state else 0.0
        ee_pos, ee_quat = get_ee_pose(arm_joint_pos)

        # grab front frame (kicks off async FFS depth), track SAM2, then reap depth/points
        ts = time.time()
        color, K, depth_scale, warp = pc['grab']()
        masks = pc['seg'].track(color)
        payload = pc['collect_depth']()

        with torch.inference_mode():
            if pc['direct_points']:
                # Orbbec: camera-frame XYZ straight from PointCloudFilter (color frame, no warp).
                xyz_m, _rgb = payload
                H, W = xyz_m.shape[:2]
                label_map = compose_label_map_torch(
                    masks, (H, W), erode=pc['erode'], device=pc['device'])
                pts = torch.as_tensor(
                    np.ascontiguousarray(xyz_m).reshape(-1, 3),
                    device=pc['device'], dtype=torch.float32)
                valid = torch.isfinite(pts[:, 2]) & (pts[:, 2] > 0)  # Orbbec: invalid depth -> z=0
                cloud, stats = B.assemble_cloud_torch(
                    pts[valid], label_map.reshape(-1)[valid], pc['T_cam_base'], ee_pos, ee_quat,
                    label_mode=True, crop_lo=pc['crop_lo'], crop_hi=pc['crop_hi'],
                    budget=pc['budget'])
                # metric color-aligned depth (m) for the eval's depth-coverage overlay
                depth = pts[:, 2].reshape(H, W).detach().cpu().numpy()
            else:
                depth = payload
                depth_t = torch.as_tensor(
                    np.ascontiguousarray(depth), device=pc['device'], dtype=torch.float32)
                if warp is None:
                    label_map = compose_label_map_torch(
                        masks, depth.shape, erode=pc['erode'], device=pc['device'])
                else:
                    masks_depth = warp_masks_to_depth_frame_torch(
                        masks, depth_t, warp['K_ir'], warp['K_color'], warp['T_ir_color'])
                    label_map = compose_label_map_torch(
                        masks_depth, depth.shape, erode=pc['erode'], device=pc['device'])
                cloud, stats = B.build_cloud_torch(
                    depth_t, K, pc['T_cam_base'], ee_pos, ee_quat, label_map=label_map,
                    depth_scale=depth_scale, crop_lo=pc['crop_lo'], crop_hi=pc['crop_hi'],
                    budget=pc['budget'])

        # PointNetPolicy.predict does np.asarray(points) -> must be host numpy (cloud is tiny)
        cloud_np = cloud.detach().cpu().numpy().astype(np.float32)
        ee_pose_vec = np.concatenate(
            [ee_pos, quat_to_axis_angle(ee_quat)]).astype(np.float32)

        # Accumulate into the recording buffer (mirrors what get_obs does for the RGB eval).
        # get_obs_pc -- NOT get_obs -- is the obs path for the point-cloud eval, so without this
        # the obs_accumulator stays empty and end_episode's min(len(obs), len(action)) collapses
        # to 0, silently dropping the whole episode (no clouds, no actions). One sample per call,
        # keyed by the grab timestamp; the segmented cloud is saved so it can be replayed/visualized.
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                {
                    'point_cloud': cloud_np[None],
                    'arm_joint_pos': arm_joint_pos.astype(np.float32)[None],
                    'gripper_pos': np.asarray([gripper_pos_raw], np.float32),
                    'end_effector_pose': ee_pose_vec[None],
                },
                np.array([ts]),
            )
        return dict(
            point_cloud=cloud_np,
            arm_joint_pos=arm_joint_pos.astype(np.float32),
            gripper_pos=np.float32(gripper_pos_raw),
            end_effector_pose=ee_pose_vec,
            cloud_stats=stats,
            color=color,
            masks=masks,
            depth=depth,             # raw depth map (FFS: left-IR frame, metres; RS: color-aligned)
            depth_scale=depth_scale, # depth units per metre (FFS: 1.0, RS: ~1000)
            K=K,                     # depth-frame intrinsics (K_ir for FFS)
            warp=warp,               # None for RS (depth==color frame); IR->color dict for FFS
            timestamp=ts,
        )

    def grab_segmented_cloud_camera_frame(self):
        """One full-resolution segmented point cloud in the CAMERA frame, no downsampling.

        Perception/quality probe (see ``eval_real_robot_pc.py --perception_test``). Runs the
        SAME grab -> SAM2 track -> warp/compose-label-map pipeline as ``get_obs_pc``, but stops
        at the camera frame and keeps EVERY valid-depth point: no base/EE transform, no crop, no
        per-class budget sampling. Background pixels keep label ``NaN`` so the full scene cloud is
        preserved alongside the segmentation -- meant for sim-vs-real perception-gap comparison.

        The points are in the same camera frame the build pipeline uses (FFS: left-IR frame;
        RealSense: color-aligned frame; Orbbec: color optical frame from PointCloudFilter) -- i.e.
        the frame ``T_cam_base`` maps to base -- so they line up with the sim camera once the same
        extrinsic is applied. NOTE the extrinsic must match the chosen camera (the Orbbec color
        frame and the D455 IR frame are different physical optical frames).

        Returns a dict:
            points:      (M, 3) float32 camera-frame xyz (metres); M = every valid-depth point.
            labels:      (M,) float32 seg label per point (SEG_LABELS values; NaN = background).
            colors:      (M, 3) uint8 per-point RGB when available (RealSense color-aligned depth,
                         or Orbbec's RGB point cloud), else None (FFS: cloud is in the IR frame).
            color:       (H, W, 3) uint8 front color frame (RGB).
            masks:       per-class SAM2 mask tensors.
            depth:       depth map / z-grid (FFS: IR frame, m; RS: color-aligned; Orbbec: color z, m).
            depth_scale: depth units per metre.
            K:           color/depth-frame intrinsics.
            warp:        None for RS/Orbbec (cloud in color frame); IR->color dict for FFS.
            timestamp:   wall-clock at grab.
        """
        assert self._pc is not None, \
            "call setup_pointcloud() before grab_segmented_cloud_camera_frame()"
        import torch
        from diffusion_policy.real_world import pointcloud_builder as B
        from diffusion_policy.real_world.pointcloud_segmenter import (
            compose_label_map_torch, warp_masks_to_depth_frame_torch)
        pc = self._pc

        # grab front frame (kicks off async FFS depth), track SAM2 (if segmenting), reap depth/points
        ts = time.time()
        color, K, depth_scale, warp = pc['grab']()
        masks = pc['seg'].track(color) if pc['seg'] is not None else None
        payload = pc['collect_depth']()

        if pc['direct_points']:
            # Orbbec: camera-frame XYZ + per-point RGB straight from PointCloudFilter (color frame).
            xyz_m, rgb_grid = payload
            H, W = xyz_m.shape[:2]
            z = xyz_m[..., 2].reshape(-1)
            valid = np.isfinite(z) & (z > 0)  # Orbbec marks invalid depth with z=0
            points_np = xyz_m.reshape(-1, 3)[valid].astype(np.float32)
            colors_np = rgb_grid.reshape(-1, 3)[valid].astype(np.uint8)
            if masks is not None:
                with torch.inference_mode():
                    label_map = compose_label_map_torch(
                        masks, (H, W), erode=pc['erode'], device=pc['device'])
                    labels_all = label_map.reshape(-1).detach().cpu().numpy().astype(np.float32)
                labels_np = labels_all[valid]
            else:  # full-cloud mode: every valid point, no labels
                labels_np = np.full(points_np.shape[0], np.nan, np.float32)
            depth = xyz_m[..., 2].astype(np.float32)  # color-aligned metric z for the vis overlay
        else:
            with torch.inference_mode():
                depth = payload
                depth_t = torch.as_tensor(
                    np.ascontiguousarray(depth), device=pc['device'], dtype=torch.float32)
                # backproject to the CAMERA frame; keep every valid-depth pixel (NO base/EE
                # transform, NO crop, NO per-class budget -- this is the un-touched cloud).
                pts_cam, pix_idx = B.backproject_torch(depth_t, K, depth_scale)
                if masks is None:  # full-cloud mode: no segmentation
                    labels = torch.full((pix_idx.numel(),), float('nan'),
                                        dtype=torch.float32, device=pts_cam.device)
                else:
                    if warp is None:
                        label_map = compose_label_map_torch(
                            masks, depth.shape, erode=pc['erode'], device=pc['device'])
                    else:
                        masks_depth = warp_masks_to_depth_frame_torch(
                            masks, depth_t, warp['K_ir'], warp['K_color'], warp['T_ir_color'])
                        label_map = compose_label_map_torch(
                            masks_depth, depth.shape, erode=pc['erode'], device=pc['device'])
                    labels = label_map.reshape(-1)[pix_idx]  # SEG_LABELS value, or NaN for background

            points_np = pts_cam.detach().cpu().numpy().astype(np.float32)
            labels_np = labels.detach().cpu().numpy().astype(np.float32)
            pix_idx_np = pix_idx.detach().cpu().numpy()

            # Per-point color only when depth is color-aligned (RealSense): then pix_idx indexes
            # the color image directly. For FFS the cloud lives in the IR frame, so the color image
            # does NOT index by pix_idx -> leave colors None (full color frame returned for vis).
            colors_np = None
            if warp is None and color is not None and color.shape[:2] == tuple(depth.shape[:2]):
                colors_np = color.reshape(-1, 3)[pix_idx_np].astype(np.uint8)

        return dict(
            points=points_np, labels=labels_np, colors=colors_np,
            color=color, masks=masks, depth=depth, depth_scale=depth_scale,
            K=np.asarray(K), warp=warp, timestamp=ts,
        )

    def setup_pose_estimation(self,
            mesh_path,
            extrinsic,
            front_serial,
            object_class='peg',
            depth_source='orbbec',
            resolution=(1280, 720),
            sam2_ckpt='orbbec/weights/sam2/sam2.1_hiera_base_plus.pt',
            sam2_cfg='configs/sam2.1/sam2.1_hiera_b+.yaml',
            prompts=None,
            est_iter=5, track_iter=2,
            fp_python=None, fp_repo=None,
            flying_pixel_mm=None,
            mock=False,
            device=None,
            async_tracking=True):
        """Stand up FoundationPose 6-DoF object-pose tracking on the front camera.

        Opens its OWN front-camera stream (color + color-aligned metric depth), grabs one
        first-frame object mask via SAM2, registers FoundationPose against ``mesh_path``, and
        leaves a worker (running in the ``foundationpose`` conda env) ready to ``track`` every
        subsequent ``get_obs_pose`` call. Standalone: needs neither ``setup_pointcloud`` nor the
        eval loop, so pose estimation can be tested on its own.

        NOTE: this opens the front camera itself. Do not also run ``setup_pointcloud`` on the same
        physical camera yet (double-open) -- sharing one grab across both is a later refactor.

        Args:
            mesh_path: object mesh (.obj, metres, object-root frame) FoundationPose tracks.
            extrinsic: (4,4) camera->base array or path (= FoundationPose T_RC); pose_base = T @ pose_cam.
            front_serial: front camera serial (advisory for Orbbec).
            object_class: label used for the interactive SAM2 prompt / which class mask to register.
            depth_source: 'orbbec' or 'realsense' (both give color-aligned metric depth).
            resolution: capture (w, h).
            prompts: {class:(pos,neg)} or (pos,neg) SAM2 clicks; None opens one interactive window.
            est_iter/track_iter: FoundationPose register / track refinement iterations.
            fp_python/fp_repo: override the foundationpose interpreter / repo (else client defaults).
            flying_pixel_mm: Orbbec-only ToF flying-pixel filter (see _open_front_orbbec).
            mock: skip FoundationPose + SAM2 (handshake/plumbing test with a dummy pose).
            async_tracking: if True (default), a background thread continuously runs
                grab->track and publishes the latest pose; ``get_obs_pose`` returns that latest
                pose instantly (never blocks on tracking), mirroring the camera ring-buffer / policy
                pattern in eval_real_robot.py. If False, ``get_obs_pose`` tracks synchronously.
        """
        from diffusion_policy.real_world.foundationpose_client import FoundationPoseClient

        if isinstance(extrinsic, str) and extrinsic.endswith('.json'):
            import json
            _d = json.load(open(extrinsic))
            _raw = _d.get('T_total_cam_simbase', _d.get('extrinsics_raw'))
            if _raw is None:
                raise KeyError(f"no 'T_total_cam_simbase'/'extrinsics_raw' in {extrinsic}")
            T_cam_base = np.asarray(_raw, np.float64).reshape(4, 4)
        elif isinstance(extrinsic, str):
            T_cam_base = np.load(extrinsic)  # legacy (4,4) .npy
        else:
            T_cam_base = np.asarray(extrinsic)
        stack = contextlib.ExitStack()

        if depth_source == 'orbbec':
            grab_ob = stack.enter_context(_open_front_orbbec(
                front_serial, resolution, flying_pixel_thresh_mm=flying_pixel_mm))

            def grab_frame():
                color, xyz_m, _rgb, K = grab_ob()
                depth_m = np.ascontiguousarray(xyz_m[..., 2], np.float32)  # color-aligned z (m)
                return color, depth_m, K
        elif depth_source == 'realsense':
            grab_rs = stack.enter_context(_open_front_realsense_depth(front_serial, resolution))

            def grab_frame():
                color, depth, K, units_per_m = grab_rs()
                depth_m = depth.astype(np.float32) / units_per_m  # raw units -> metres
                return color, depth_m, K
        else:
            stack.close()
            raise ValueError(
                f"pose-estimation depth_source must be 'orbbec' or 'realsense', got {depth_source!r}")

        try:
            color, depth_m, K = grab_frame()  # frame 0
            seg = None
            if mock:
                mask = (depth_m > 0)  # dummy ROI for the handshake test
            else:
                from diffusion_policy.real_world.pointcloud_segmenter import (
                    PointCloudSegmenter, pick_prompts_interactive)
                if prompts is None:
                    prompts = pick_prompts_interactive(color, class_names=(object_class,))
                pos, neg = prompts[object_class] if isinstance(prompts, dict) else prompts
                seg = PointCloudSegmenter(sam2_ckpt, sam2_cfg, device=device)
                mask = seg.mask(color, pos, neg)
            if mask is None or not np.any(mask):
                raise RuntimeError("setup_pose_estimation: empty object mask (need >=1 positive click)")

            client_kw = dict(mesh_path=mesh_path, est_iter=est_iter,
                             track_iter=track_iter, mock=mock)
            if fp_python is not None:
                client_kw['fp_python'] = fp_python
            if fp_repo is not None:
                client_kw['fp_repo'] = fp_repo
            client = FoundationPoseClient(**client_kw)
            client.start()
            client.wait_ready()
            pose_cam = client.register(color, depth_m, K, np.asarray(mask, bool))
        except Exception:
            stack.close()
            raise

        self._pose_stack = stack
        self._pose = dict(
            client=client, grab_frame=grab_frame, T_cam_base=T_cam_base,
            object_class=object_class, seg=seg,
            lock=threading.Lock(), stop_event=threading.Event(), thread=None, errors=0,
        )
        # seed the latest-pose buffer with the registration result (frame 0)
        self._pose['latest'] = self._make_pose_obs(pose_cam, K=K, color=color, depth=depth_m)
        if async_tracking:
            th = threading.Thread(target=self._pose_loop, name='fp-pose-loop', daemon=True)
            self._pose['thread'] = th
            th.start()
        pos0 = self._pose['latest']['object_pos']
        print(f"[RealEnv] FoundationPose up: object={object_class}, depth_source={depth_source}, "
              f"async={async_tracking}, mock={mock}, registered pos(base)={pos0.round(4).tolist()}")
        return self.get_obs_pose()

    def _make_pose_obs(self, pose_cam, K=None, color=None, depth=None, ts=None) -> dict:
        """Build the object-pose obs dict (camera + base frame) from a (4,4) object-in-camera."""
        pose_base = self._pose['T_cam_base'] @ pose_cam
        return dict(
            object_pose_cam=pose_cam.astype(np.float32),
            object_pose_base=pose_base.astype(np.float32),
            object_pos=pose_base[:3, 3].astype(np.float32),
            object_quat_wxyz=_rotmat_to_quat_wxyz(pose_base[:3, :3]),
            K=K, color=color, depth=depth,
            timestamp=ts if ts is not None else time.time(),
        )

    def _pose_loop(self):
        """Background producer: continuously grab -> track -> publish the latest object pose.

        Runs in its own thread (started by ``setup_pose_estimation(async_tracking=True)``). The
        heavy FoundationPose compute happens in the worker subprocess, so this thread mostly waits
        on shared memory; ``get_obs_pose`` reads the published latest without blocking. Mirrors the
        camera-worker/ring-buffer decoupling in eval_real_robot.py.
        """
        ps = self._pose
        while not ps['stop_event'].is_set():
            try:
                color, depth_m, K = ps['grab_frame']()
                pose_cam = ps['client'].track(color, depth_m, K)
            except Exception as e:  # transient grab/track hiccup: keep last pose, keep looping
                ps['errors'] += 1
                if ps['errors'] <= 3 or ps['errors'] % 100 == 0:
                    print(f"[RealEnv] pose loop error #{ps['errors']}: {e}")
                continue
            obs = self._make_pose_obs(pose_cam, K=K, color=color, depth=depth_m)
            with ps['lock']:
                ps['latest'] = obs

    def get_obs_pose(self) -> dict:
        """Latest 6-DoF object pose from FoundationPose (call ``setup_pose_estimation`` first).

        Async mode (default): returns the freshest pose published by the background tracking
        thread -- instant, non-blocking, decoupled from the tracking rate (like the camera
        ring-buffer the RGB policy reads). Sync mode: grabs a frame and tracks inline.

        Returns a dict:
            object_pose_cam:   (4,4) object-in-camera.
            object_pose_base:  (4,4) object-in-base  (= T_cam_base @ object_pose_cam).
            object_pos:        (3,) float32 object position in base frame (m).
            object_quat_wxyz:  (4,) float32 object orientation in base frame.
            K:                 (3,3) color intrinsics used.
            color, depth:      the frame fed to FoundationPose (RGB, metric depth).
            timestamp:         float wall-clock when this pose was produced (check for staleness).
        """
        assert self._pose is not None, "call setup_pose_estimation() before get_obs_pose()"
        ps = self._pose
        if ps.get('thread') is not None:
            # async: return the latest pose published by the background loop (never block/track here)
            with ps['lock']:
                return dict(ps['latest'])
        # sync fallback: track inline
        ts = time.time()
        color, depth_m, K = ps['grab_frame']()
        pose_cam = ps['client'].track(color, depth_m, K)
        ps['latest'] = self._make_pose_obs(pose_cam, K=K, color=color, depth=depth_m, ts=ts)
        return dict(ps['latest'])

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray,
            stages: Optional[np.ndarray]=None,
            obs_actions: Optional[np.ndarray]=None):
        """
        Execute unified robot actions using OSC torque control.

        last_arm_action / last_gripper_action in get_obs() come from the action buffer.
        If obs_actions is provided, that (raw/pre-scale) is stored in the buffer and used
        for recording; otherwise actions is stored.

        Args:
            actions: Unified robot actions (shape: N x 7) to execute:
                - action_mode='joint':  actions[:, :6] = target joint positions (rad)
                - action_mode='cartesian': actions[:, :6] = absolute EE target
                    [px, py, pz, ax, ay, az] (position + axis-angle orientation)
                - actions[:, 6] = Gripper position (<0=closed, >=0=open)
            timestamps: Action timestamps
            stages: Optional stage information
            obs_actions: Optional (shape: N x (arm_action_dim + 1), i.e. the raw policy
                action: arm dims + 1 gripper). If set, stored in action buffer and
                accumulators so last_arm_action in obs is pre-scale; actions are still executed.
        """
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # Validate action shape
        if actions.shape[-1] != 7:
            raise ValueError(f"Actions must have 7 dimensions (6 arm + 1 gripper), got shape {actions.shape}")
        
        # Separate arm and gripper actions
        arm_actions = actions[:, :6]
        # Convert gripper action: <0 is closed, >=0 is open
        gripper_actions = actions[:, 6:7] < 0

        # Filter to future actions only
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_arm_actions = arm_actions[is_new]
        new_gripper_actions = gripper_actions[is_new]
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # Execute actions via OSC
        for i in range(len(new_arm_actions)):
            if self.action_mode == 'cartesian':
                target_pos = new_arm_actions[i, :3]
                target_quat = axis_angle_to_quat(new_arm_actions[i, 3:6])
                self.robot.cartesian_osc_control(
                    target_pos=target_pos,
                    target_quat=target_quat,
                    close_gripper=new_gripper_actions[i]
                )
            else:
                self.robot.joint_torque_control(
                    target_joints=new_arm_actions[i],
                    close_gripper=new_gripper_actions[i]
                )
        
        # Store pre-scale (obs_actions) in buffer/accumulators when provided
        to_store = new_actions
        if obs_actions is not None:
            obs_actions = np.array(obs_actions)
            expected_obs_dim = self.arm_action_dim + 1
            if obs_actions.shape[-1] != expected_obs_dim:
                raise ValueError(
                    f"obs_actions must have {expected_obs_dim} dimensions "
                    f"({self.arm_action_dim} arm + 1 gripper), got shape {obs_actions.shape}")
            if obs_actions.shape[0] == actions.shape[0]:
                to_store = obs_actions[is_new]
            else:
                to_store = obs_actions[-len(new_actions):]
            assert len(to_store) == len(new_actions), "obs_actions length must match executed actions"

        if self.action_accumulator is not None:
            self.action_accumulator.put(
                to_store,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
        if self.action_buffer is not None:
            for action in to_store:
                self.action_buffer.append(action)

        # Canonical last raw action: the most recent action the policy issued this step. Updated
        # on EVERY call (even if scheduling filtered it out of execution) so prev_action never
        # stalls. Prefer obs_actions (raw/pre-scale); fall back to the absolute target actions.
        # When nothing was is_new, also advance the rolling history so it never stalls either.
        raw_full = obs_actions if obs_actions is not None else actions
        if len(raw_full) > 0:
            self.last_action = np.asarray(raw_full[-1], dtype=np.float32).copy()
            if self.action_buffer is not None and len(to_store) == 0:
                self.action_buffer.append(self.last_action)

    def get_last_action(self):
        """Canonical last raw action (arm_action_dim + 1) the policy issued, or zeros if none yet."""
        if self.last_action is None:
            return np.zeros(self.arm_action_dim + 1, dtype=np.float32)
        return self.last_action.copy()

    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # Reset the canonical last action to zeros each episode (matches sim last_action reset).
        self.last_action = np.zeros(self.arm_action_dim + 1, dtype=np.float32)

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            try:
                if n_steps > 0:
                    episode = dict()
                    episode['timestamp'] = obs_timestamps[:n_steps]
                    episode['action'] = actions[:n_steps]
                    episode['stage'] = stages[:n_steps]
                    for key, value in obs_data.items():
                        episode[key] = value[:n_steps]
                    self.replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = self.replay_buffer.n_episodes - 1
                    print(f'Episode {episode_id} saved!')
            except Exception as e:
                # Never let a failed save wedge the session: the accumulators are cleared
                # in `finally`, so callers (and stop()/__exit__) can still shut down cleanly.
                print(f'[RealEnv] Failed to save episode, discarding it: {e}')
            finally:
                self.obs_accumulator = None
                self.action_accumulator = None
                self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
