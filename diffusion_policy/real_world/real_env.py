from typing import Optional
import contextlib
import pathlib
import numpy as np
import time
import shutil
import math
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
        
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
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

        custom_init_joints = np.array([0.020015, -1.39426, 2.13054, -2.246, -1.618852138519287, -0.087])
        
        # Handle joint initialization
        j_init = None
        if init_joints:
            if custom_init_joints is not None:
                # Use custom initial joint positions if provided
                j_init = np.array(custom_init_joints)
                print(f"Using custom initial joint positions: {j_init}")
            else:
                # Use default initial joint positions
                j_init = np.array([16.85, -79.74, 99.80, -114.68, -91.09, 20.43]) / 180 * np.pi
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
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        # No-timestamp action buffer
        rolling_action_buffer = self.action_buffer = deque(maxlen=self.n_obs_steps) if rolling_action_buffer else None

        self.start_time = None

        # Point-cloud obs pipeline (front D455 stereo/FFS + SAM2 streaming). Set up lazily by
        # ``setup_pointcloud`` and torn down in ``stop``; ``None`` until then so plain RGB/depth
        # eval is unaffected. See ``get_obs_pc`` / debug_pointcloud.py for the same pipeline.
        self._pc_stack = None
        self._pc = None
    
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
        self.end_episode()
        if self._pc_stack is not None:
            self._pc_stack.close()  # close front stereo/FFS stream + SAM2 worker
            self._pc_stack = None
            self._pc = None
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
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key: 3-camera setup has front/side/wrist; 2-camera skips front
            n_cams = self.realsense.n_cameras
            if n_cams >= 3:
                cam_names = ['front_rgb', 'side_rgb', 'wrist_rgb']
            else:
                cam_names = ['side_rgb', 'wrist_rgb']
            if camera_idx < len(cam_names):
                camera_obs[cam_names[camera_idx]] = value['color'][this_idxs]
        
        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # last_arm_action / last_gripper_action: from action buffer (pre-scale when exec_actions called with obs_actions)
        # Buffer rows are [arm_action_dim arm dims, 1 gripper dim]; width adapts to the policy.
        arm_dim = self.arm_action_dim
        act_dim = arm_dim + 1
        last_actions = dict()
        if self.action_buffer is not None and len(self.action_buffer) > 0:
            last_actions_raw = np.zeros((self.n_obs_steps, act_dim), dtype=np.float32)
            actions = np.array(self.action_buffer)

            # overlay the buffer in
            last_actions_raw[:actions.shape[0], :] = actions

            last_actions = {
                'last_arm_action': last_actions_raw[:, :arm_dim],
                'last_gripper_action': last_actions_raw[:, arm_dim:act_dim]
            }
        else:
            # values = self.robot.get_state()['ActualQ']
            obs_array = np.zeros((self.n_obs_steps, act_dim), dtype=np.float32)
            last_actions = {
                'last_arm_action': obs_array[:, :arm_dim],
                'last_gripper_action': np.zeros((self.n_obs_steps, 1))
            }
        
        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(last_actions)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    # ========= point-cloud obs API ===========
    def setup_pointcloud(self,
            front_serial,
            extrinsic,
            depth_source='ffs',
            resolution=(1280, 720),
            sam2_ckpt='orbbec/weights/sam2/sam2.1_hiera_base_plus.pt',
            sam2_cfg='configs/sam2.1/sam2.1_hiera_b+.yaml',
            erode=3,
            crop_lo=None, crop_hi=None,
            budget=None,
            prompts=None,
            ffs_mock=False,
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
            ffs_mock: ramp depth for plumbing tests without FFS weights.
            device: torch device for the on-GPU cloud build (defaults to the SAM2 device).
        """
        from diffusion_policy.real_world.pointcloud_segmenter import (
            StreamingSegmenter, pick_prompts_interactive)

        T_cam_base = np.load(extrinsic) if isinstance(extrinsic, str) else np.asarray(extrinsic)
        stack = contextlib.ExitStack()

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
        else:
            stack.close()
            raise ValueError(f"depth_source must be 'ffs' or 'realsense', got {depth_source!r}")

        try:
            color, K, depth_scale, warp = grab()
            if prompts is None:
                prompts = pick_prompts_interactive(color)
            if not prompts:
                raise RuntimeError("setup_pointcloud: no SAM2 prompts (need >=1 class clicked)")
            seg = StreamingSegmenter(sam2_ckpt, sam2_cfg, device=device)
            masks = seg.start(color, prompts)  # seed tracker on frame 0
            _ = collect_depth()  # reap (+ discard) the frame-0 depth submitted by grab()
        except Exception:
            stack.close()
            raise

        self._pc_stack = stack
        self._pc = dict(
            grab=grab, collect_depth=collect_depth, seg=seg, device=seg.device,
            T_cam_base=T_cam_base, erode=erode, crop_lo=crop_lo, crop_hi=crop_hi,
            budget=budget, depth_source=depth_source,
        )
        print(f"[RealEnv] point-cloud pipeline up: depth_source={depth_source}, "
              f"classes={list(prompts)}, device={seg.device}")

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

        # grab front frame (kicks off async FFS depth), track SAM2, then reap depth
        ts = time.time()
        color, K, depth_scale, warp = pc['grab']()
        masks = pc['seg'].track(color)
        depth = pc['collect_depth']()

        with torch.inference_mode():
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
        return dict(
            point_cloud=cloud_np,
            arm_joint_pos=arm_joint_pos.astype(np.float32),
            gripper_pos=np.float32(gripper_pos_raw),
            end_effector_pose=np.concatenate(
                [ee_pos, quat_to_axis_angle(ee_quat)]).astype(np.float32),
            cloud_stats=stats,
            color=color,
            masks=masks,
            timestamp=ts,
        )

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

    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

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

