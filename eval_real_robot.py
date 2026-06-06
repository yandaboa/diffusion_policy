"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
Press "R" to reset robot to initial position and start new trajectory.

The episode auto-terminates when the EE rises above --z_terminate (default 0.4 m).
On any auto-termination you are prompted to label the episode: press "S" for
success or "F" for fail. Labels are written to <save_dir>/eval_results.json.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import json
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.transformer_image_policy import TransformerImagePolicy

# Add imageio import for video saving
import imageio
from scipy.spatial.transform import Rotation as R

# Calibrated FK matching simulation (wrist_3_link in REP-103 base_link frame)
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)


def compute_binary_contact(tcp_force, threshold):
    """Compute binary contact from TCP force/torque sensor.
    
    Mirrors sim's binary_force_contact: ||F[:3]|| > threshold -> 1.0 else 0.0.
    
    Args:
        tcp_force: (n_obs_steps, 6) wrench [Fx,Fy,Fz,Tx,Ty,Tz] from UR F/T sensor
        threshold: Force norm threshold in Newtons
    Returns:
        binary_contact: (n_obs_steps, 1) float32
    """
    force_norm = np.linalg.norm(tcp_force[:, :3], axis=-1)
    contact = (force_norm > threshold).astype(np.float32)
    return contact[:, None]


def build_policy_input(obs_dict_np, obs_history, policy, device, max_history):
    """Construct the obs dict to feed `policy.predict_action`.

    For ``TransformerImagePolicy`` (ASTEROID in-context model), accumulates a
    rolling per-episode history of observations and emits a (1, T, *) sequence
    tensor + (1, T) ``attention_mask``. Caller must clear ``obs_history`` on
    every ``policy.reset()`` to start a fresh in-context trajectory.

    For all other policies, falls back to the standard single-step
    ``unsqueeze(0)`` path that yields (1, n_obs_steps, *) per key.
    """
    if isinstance(policy, TransformerImagePolicy):
        for k, v in obs_dict_np.items():
            # v shape: (n_obs_steps, *) from get_real_obs_dict; keep most recent frame.
            obs_history.setdefault(k, []).append(v[-1])
            if len(obs_history[k]) > max_history:
                obs_history[k].pop(0)
        T = len(next(iter(obs_history.values())))
        seq = {}
        for k, hist in obs_history.items():
            arr = np.stack(hist, axis=0)[None]  # (1, T, *)
            seq[k] = torch.from_numpy(arr).to(device)
        seq['attention_mask'] = torch.ones((1, T), dtype=torch.long, device=device)
        return seq
    return dict_apply(obs_dict_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))


def compute_calibrated_ee_pose(joint_positions):
    """Compute EE pose using calibrated FK to wrist_3_link (matching simulation).
    
    Uses calibrated URDF parameters with 180deg Z base rotation (REP-103 frame).
    Returns [x, y, z, rx, ry, rz] where rotation is axis-angle, matching sim's
    target_asset_pose_in_root_asset_frame with rotation_repr='axis_angle'.
    
    Args:
        joint_positions: (n_obs_steps, 6) joint angles in radians
    Returns:
        ee_poses: (n_obs_steps, 6) [x, y, z, rx, ry, rz]
    """
    n_steps = joint_positions.shape[0]
    ee_poses = np.zeros((n_steps, 6), dtype=np.float32)
    for t in range(n_steps):
        pos, quat = get_ee_pose(joint_positions[t])
        axis_angle = quat_to_axis_angle(quat)
        ee_poses[t, :3] = pos
        ee_poses[t, 3:] = axis_angle
    return ee_poses


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, 
              help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, 
              help="UR5's IP address e.g. 192.168.1.10")
@click.option('--match_dataset', '-m', default=None, 
              help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, 
              help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, 
              help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, 
              help="Whether to initialize robot joint configuration in the "
                   "beginning.")
@click.option('--steps_per_inference', '-si', default=1, type=int, 
              help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=20,
              help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, 
              help="Control frequency in Hz.")
@click.option('--save_video', is_flag=True, default=False,
              help='Save video of concatenated camera views.')
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise added to raw arm actions (pre-scale).')
@click.option('--contact_threshold', default=5.0, type=float,
              help='Force norm (N) threshold for binary_contact obs. '
                   'Sim uses 25.0 on joint wrench; real F/T sensor differs.')
@click.option('--collect_sysid', default=None, type=str,
              help='Save on-policy sysid data to .pt file (joint traj + OSC targets)')
@click.option('--plot_gripper', is_flag=True, default=False,
              help='Save a per-episode plot of the gripper_pos observation (and the '
                   'commanded gripper) over time as a sanity check.')
@click.option('--z_terminate', default=0.4, type=float,
              help='Auto-terminate the episode when the EE z height (REP-103 base '
                   'frame, m) exceeds this value. After termination the user is '
                   "prompted to label the episode ('s'=success, 'f'=fail).")
@click.option('--input_res', default='1280x720', type=str,
              help='Camera capture resolution as WxH. Defaults to 1280x720, '
                   'the highest resolution common to D415/D435/D455 at 30fps. '
                   'Policy obs resolution is set by the checkpoint independently.')
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints,
         steps_per_inference, max_duration,
         frequency, save_video, action_noise, contact_threshold,
         collect_sysid, plot_gripper, z_terminate, input_res):
    # Parse camera capture resolution
    capture_w, capture_h = (int(x) for x in input_res.lower().split('x'))
    capture_resolution = (capture_w, capture_h)
    print(f"Camera capture resolution: {capture_resolution}")

    # Per-axis Cartesian scale matching the sim action config (raw policy output -> meters/rad).
    #   - Legacy 6-DOF action [x,y,z,rx,ry,rz]: full pose delta.
    #   - Position-only action [x,y,z]: xyz delta only; orientation is left uncommanded
    #     (RelCartesianOSCPositionAction, scale_xyz_axisangle=(0.02, 0.02, 0.02, ...)).
    # The active scale is selected below once the checkpoint's action dim is known.
    CARTESIAN_SCALE_6DOF = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    CARTESIAN_SCALE_POSONLY = np.array([0.01, 0.01, 0.002])

    # Sysid data collection state
    sysid_records = []  # list of (joint_pos, target_pos, target_quat)

    def save_sysid_data():
        """Save 10Hz policy waypoints for 500Hz replay via test_real_ur5e_osc_cube.py --replay_eval."""
        if not collect_sysid or len(sysid_records) == 0:
            return
        import torch as _torch
        jp = np.array([r[0] for r in sysid_records])
        wp_pos = np.array([r[1] for r in sysid_records])
        wp_quat = np.array([r[2] for r in sysid_records])
        n = len(sysid_records)
        _torch.save({
            "joint_positions": _torch.tensor(jp, dtype=_torch.float32),
            "initial_joint_pos": _torch.tensor(jp[0], dtype=_torch.float32),
            "waypoint_step_indices": _torch.arange(n, dtype=_torch.long),
            "waypoint_target_pos": _torch.tensor(wp_pos, dtype=_torch.float32),
            "waypoint_target_quat": _torch.tensor(wp_quat, dtype=_torch.float32),
            "dt": dt,
        }, collect_sysid)
        print(f"\nSaved sysid data ({n} policy steps at {frequency}Hz) to: {collect_sysid}")

    # Gripper-pos sanity-check plotting state.
    # Each record is (t_rel_s, gripper_pos_obs, commanded_gripper) for the current episode.
    gripper_pos_records = []
    _gripper_plot_count = [0]  # mutable so the closure can advance the per-episode file index

    def save_gripper_plot():
        """Save the current episode's gripper_pos observation (and commanded gripper) vs time."""
        if not plot_gripper or len(gripper_pos_records) == 0:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        arr = np.array(gripper_pos_records, dtype=np.float32)
        idx = _gripper_plot_count[0]
        _gripper_plot_count[0] += 1
        plot_path = pathlib.Path(output) / f'gripper_pos_ep_{idx:03d}.png'
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(arr[:, 0], arr[:, 1], color='C0', label='gripper_pos (obs)')
        ax.plot(arr[:, 0], arr[:, 2], color='C1', alpha=0.6,
                drawstyle='steps-post', label='commanded (>0=open)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('gripper')
        ax.set_title(f'Gripper pos observation — episode {idx}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\nSaved gripper plot ({len(arr)} steps) to: {plot_path}")

    # Episode success/fail labels collected after auto-termination.
    episode_results = []  # list of dicts: {'episode', 'result', 'z', 't'}

    def prompt_success_fail(episode_id, ee_z):
        """Block (via the OpenCV window) until the user labels the episode.

        Press 's' = SUCCESS, 'f' = FAIL. Returns the bool result and appends a
        record to ``episode_results``, persisted to ``output/eval_results.json``.
        """
        print(f"Episode {episode_id} terminated (EE z={ee_z:.3f}). "
              "Label it: 's'=SUCCESS, 'f'=FAIL ...")
        result = None
        while result is None:
            canvas = np.zeros((200, 700, 3), dtype=np.uint8)
            cv2.putText(canvas, f"EPISODE {episode_id} TERMINATED (z={ee_z:.3f})",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "'s' = SUCCESS", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, "'f' = FAIL", (15, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Policy Control', canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('s'):
                result = True
            elif key == ord('f'):
                result = False
        print(f"  -> {'SUCCESS' if result else 'FAIL'}")
        episode_results.append({
            'episode': int(episode_id),
            'result': 'success' if result else 'fail',
            'z': float(ee_z),
            't': time.time(),
        })
        n_succ = sum(r['result'] == 'success' for r in episode_results)
        n_tot = len(episode_results)
        results_path = pathlib.Path(output) / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'n_episodes': n_tot,
                'n_success': n_succ,
                'success_rate': n_succ / n_tot,
                'episodes': episode_results,
            }, f, indent=2)
        print(f"  [{n_succ}/{n_tot} success] saved to {results_path}")
        return result

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Detect which cameras are physically connected and filter accordingly.
    # RealEnv maps camera index 0→front_rgb, 1→side_rgb, 2→wrist_rgb, so
    # dropping a serial from the list simply omits that key from obs.
    import pyrealsense2 as _rs
    _connected = {
        d.get_info(_rs.camera_info.serial_number)
        for d in _rs.context().devices
        if d.get_info(_rs.camera_info.name).lower() != 'platform camera'
    }
    _all_cameras = [
        # ('215122255213', json.load(open("diffusion_policy/real_world/realsense_config/455_front.json"))),
        ('832112070487', json.load(open("diffusion_policy/real_world/realsense_config/435_side.json"))),
        # ('746112060198', json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json"))),
    ]
    _active = [(s, c) for s, c in _all_cameras if s in _connected]
    _missing = [s for s, _ in _all_cameras if s not in _connected]
    if _missing:
        print(f"Warning: cameras not connected, skipping: {_missing}")
    camera_serial_numbers = [s for s, _ in _active]
    configs = [c for _, c in _active]
    print(f"Active cameras: {camera_serial_numbers}")

    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    if 'extra_randomizations' in cfg['policy']['obs_encoder']:
        cfg['policy']['obs_encoder']['extra_randomizations'] = []
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(device)

    # Diffusion-specific overrides (no-op for MLP / Transformer policies)
    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # In-context (TransformerImagePolicy) needs per-episode obs history.
    # Cap at training horizon so we never overflow GPT2 positional embeddings.
    is_in_context = isinstance(policy, TransformerImagePolicy)
    obs_history = {}
    in_context_max_history = int(cfg.get('horizon', 256))
    if is_in_context:
        print(f"In-context policy detected; obs history capped at {in_context_max_history} steps.")

    # setup experiments
    dt = 1/frequency
    obs_res = get_real_obs_resolution(cfg['task']['shape_meta'])
    if obs_res is None:
        obs_res = (640, 480)  # no image obs in checkpoint; use capture resolution for recording
    n_obs_steps = cfg['n_obs_steps']
    n_action_steps = cfg['n_action_steps']
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference: ", steps_per_inference)
    print("n_action_steps: ", n_action_steps)

    # Action-space layout from the checkpoint. The last dim is the binary gripper;
    # the remaining arm dims are either 6 (full Cartesian pose delta) or 3 (position-only).
    action_dim = int(cfg['shape_meta']['action']['shape'][0])
    arm_action_dim = action_dim - 1
    if arm_action_dim not in (3, 6):
        raise ValueError(
            f"Unsupported action_dim={action_dim}; expected 4 (3 arm + gripper) "
            f"or 7 (6 arm + gripper).")
    position_only = (arm_action_dim == 3)
    CARTESIAN_SCALE = CARTESIAN_SCALE_POSONLY if position_only else CARTESIAN_SCALE_6DOF
    print(f"action_dim: {action_dim} ({arm_action_dim} arm + 1 gripper), "
          f"position_only={position_only}")
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            rolling_action_buffer=True,
            action_mode='cartesian',
            arm_action_dim=arm_action_dim,
            camera_serial_numbers=camera_serial_numbers,
            camera_configs=configs,
            video_capture_resolution=capture_resolution,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            
            cv2.setNumThreads(1)

            print("Waiting for realsense")
            time.sleep(5.0)

            print("Warming up policy inference")
            print(f"Contact threshold: {contact_threshold} N")
            obs = env.get_obs()
            # Override EE pose with calibrated FK (wrist_3_link in REP-103 frame)
            obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])
            if 'tcp_force' in obs:
                obs['binary_contact'] = compute_binary_contact(obs['tcp_force'], contact_threshold)

            with torch.no_grad():
                policy.reset()
                obs_history.clear()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg['shape_meta'])
                obs_dict = build_policy_input(
                    obs_dict_np, obs_history, policy, device,
                    in_context_max_history)
                try:
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().to('cpu').numpy()
                    del result
                except Exception as e:
                    print(e)
                    # Handle case where result might not be defined
                    if 'result' in locals():
                        del result

            print('Ready!')
            time.sleep(1.0)
            
            # Initialize video recording if enabled
            video_fps = int(frequency)
            episode_video_writer = None
            long_video_writer = None
            if save_video:
                long_video_path = pathlib.Path(output) / 'policy_cameras_full.mp4'
                long_video_writer = imageio.get_writer(
                    str(long_video_path), fps=video_fps, codec='libx264',
                    output_params=['-crf', '21', '-preset', 'fast'])
                print(f"Video recording enabled at {video_fps} fps")
                print(f"  Continuous video: {long_video_path}")

            actions = []
            gripper_open_steps_remaining = 0
            GRIPPER_OPEN_DURATION = 5  # timesteps to hold gripper open when 'g' pressed
            # Stuck detection: if robot doesn't move for this long, open gripper to get unstuck
            STUCK_WINDOW_S = 2.0
            STUCK_JOINT_THRESHOLD_RAD = 0.002  # ~0.1 deg max movement per joint over window
            STUCK_GRIPPER_OPEN_STEPS = int(frequency)  # 1 s open at control freq
            stuck_buffer = []  # list of (t, joint_pos)
            
            while True:
                # ========== policy control loop ==============
                try:
                    # Reset robot to its initial position between episodes so every
                    # episode starts from the same home pose (also covers the 's'-stop
                    # path, which otherwise restarts wherever the previous run ended).
                    print('Resetting robot to initial position...')
                    env.robot.reset_to_initial_position()
                    time.sleep(5.0)
                    print('Reset complete.')

                    # start episode
                    policy.reset()
                    obs_history.clear()
                    gripper_open_steps_remaining = 0
                    sysid_records.clear()
                    gripper_pos_records.clear()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start, time_func=time.time)
                    print("Started!")
                    stuck_buffer.clear()
                    if save_video:
                        episode_id_start = getattr(env.replay_buffer, 'n_episodes', 0)
                        ep_video_path = pathlib.Path(output) / f'policy_cameras_ep_{episode_id_start:03d}.mp4'
                        episode_video_writer = imageio.get_writer(
                            str(ep_video_path), fps=video_fps, codec='libx264',
                            output_params=['-crf', '21', '-preset', 'fast'])
                        print(f"  Episode video: {ep_video_path}")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        # Override EE pose with calibrated FK (wrist_3_link in REP-103 frame)
                        obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])
                        if 'tcp_force' in obs:
                            obs['binary_contact'] = compute_binary_contact(obs['tcp_force'], contact_threshold)

                        # Capture frames for video if enabled (streamed to disk in real-time)
                        if save_video:
                            # camera_names = ['front_rgb', 'side_rgb', 'wrist_rgb']
                            camera_names = ['front_rgb', 'side_rgb']
                            imgs = []
                            for cam_name in camera_names:
                                if cam_name in obs:
                                    img = obs[cam_name][-1]
                                    if img.dtype == np.float32 or img.dtype == np.float64:
                                        img = (img * 255).clip(0, 255).astype(np.uint8)
                                    imgs.append(img)
                            if len(imgs) == 3:
                                frame = np.concatenate(imgs, axis=1)
                                if episode_video_writer is not None:
                                    episode_video_writer.append_data(frame)
                                if long_video_writer is not None:
                                    long_video_writer.append_data(frame)

                        # run inference
                        with torch.no_grad():
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg['shape_meta']
                            )
                            obs_dict = build_policy_input(
                                obs_dict_np, obs_history, policy, device,
                                in_context_max_history)
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0:1].detach().to('cpu').numpy()
                        
                        # action shape: (N, action_dim) where [:, :arm_action_dim] is the
                        # Cartesian delta (xyz[+rpy]) and the last dim is the binary gripper.
                        raw_arm_action = action[:, :arm_action_dim]  # Raw network output (pre-scale)
                        if action_noise > 0:
                            raw_arm_action = raw_arm_action + np.random.randn(*raw_arm_action.shape) * action_noise
                        gripper_actions = action[:, arm_action_dim:arm_action_dim+1]

                        # Stuck detection: if robot barely moved for STUCK_WINDOW_S, open gripper to get unstuck
                        if gripper_open_steps_remaining == 0:
                            t_now = time.monotonic()
                            stuck_buffer.append((t_now, obs['arm_joint_pos'][-1].copy()))
                            # keep only last STUCK_WINDOW_S
                            while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                stuck_buffer.pop(0)
                            if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                jps = np.array([b[1] for b in stuck_buffer])
                                range_per_joint = jps.max(axis=0) - jps.min(axis=0)
                                if np.max(range_per_joint) < STUCK_JOINT_THRESHOLD_RAD:
                                    gripper_open_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                    stuck_buffer.clear()
                                    print("[Stuck detection] No movement for 2s, opening gripper")

                        # Gripper open macro: override policy gripper command
                        if gripper_open_steps_remaining > 0:
                            gripper_actions = np.ones_like(gripper_actions)  # >0 = open
                            gripper_open_steps_remaining -= 1
                            if gripper_open_steps_remaining == 0:
                                print("[Gripper macro] done, returning to policy control")

                        # Sanity-check log: gripper_pos observation vs commanded gripper.
                        if plot_gripper and 'gripper_pos' in obs:
                            gripper_pos_records.append((
                                time.monotonic() - t_start,
                                float(np.asarray(obs['gripper_pos'][-1]).reshape(-1)[0]),
                                float(gripper_actions[0, 0]),
                            ))

                        raw_actions = np.concatenate([raw_arm_action, gripper_actions], axis=1)  # for last_arm_action obs

                        # Cartesian OSC: scale delta, compute absolute target from observed pose
                        scaled_delta = raw_arm_action * CARTESIAN_SCALE
                        obs_jp = obs['arm_joint_pos'][-1]
                        obs_pos, obs_quat = get_ee_pose(obs_jp)
                        if position_only:
                            # Policy commands xyz only; orientation is left uncommanded so the
                            # target orientation tracks the current EE orientation (no rotation
                            # error accumulates -- mirrors RelCartesianOSCPositionAction in sim).
                            tgt_pos = obs_pos + scaled_delta[0]
                            tgt_quat = obs_quat
                        else:
                            tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled_delta[0])
                        tgt_aa = quat_to_axis_angle(tgt_quat)
                        abs_target = np.concatenate([tgt_pos, tgt_aa])[None]  # (1, 6)
                        target_actions = np.concatenate([abs_target, gripper_actions], axis=1)

                        if collect_sysid:
                            sysid_records.append((obs_jp.copy(), tgt_pos.copy(), tgt_quat.copy()))

                        # deal with timing
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            target_actions = target_actions[[-1]]
                            raw_actions = raw_actions[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            action_timestamps = np.array([action_timestamp])
                        else:
                            target_actions = target_actions[is_new]
                            raw_actions = raw_actions[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # Execute actions; store raw (pre-scale) in buffer so last_arm_action is raw
                        actions.append(target_actions)
                        env.exec_actions(
                            actions=target_actions[:n_action_steps],
                            timestamps=action_timestamps[:n_action_steps],
                            obs_actions=raw_actions[:n_action_steps]
                        )

                        # Visualize camera feed for key detection
                        episode_id = env.replay_buffer.n_episodes
                        camera_key = 'side_rgb'
                        if camera_key in obs:
                            vis_img = obs[camera_key][-1]
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('Policy Control', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening gripper for {GRIPPER_OPEN_DURATION} steps")
                        elif key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            save_sysid_data()
                            save_gripper_plot()
                            env.end_episode()
                            print('Stopped.')
                            break
                        elif key_stroke == ord('r'):
                            # Reset robot and start new trajectory
                            save_sysid_data()
                            save_gripper_plot()
                            sysid_records.clear()
                            gripper_pos_records.clear()
                            stuck_buffer.clear()
                            print('Resetting robot for new trajectory...')
                            env.end_episode()
                            
                            # Close per-episode video writer
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            
                            # Reset policy state
                            policy.reset()
                            obs_history.clear()
                            
                            # Move robot to initial position
                            env.robot.reset_to_initial_position()
                            
                            # Wait a moment for robot to settle
                            time.sleep(5.0)
                            
                            # Start new episode
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            precise_wait(eval_t_start, time_func=time.time)
                            
                            # Reset iteration counter
                            iter_idx = 0
                            term_area_start_timestamp = float('inf')
                            
                            print('Robot reset complete! Starting new trajectory.')
                            continue

                        # auto termination
                        terminate = False
                        ee_z = float(obs_pos[2])  # EE height (REP-103 base frame)
                        if ee_z > z_terminate:
                            terminate = True
                            print(f'Terminated: EE z={ee_z:.3f} > {z_terminate:.3f}')
                        elif time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        # term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        # curr_pose = obs['robot_eef_pose'][-1]
                        # dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        # if dist < 0.03:
                        #     # in termination area
                        #     curr_timestamp = obs['timestamp'][-1]
                        #     if term_area_start_timestamp > curr_timestamp:
                        #         term_area_start_timestamp = curr_timestamp
                        #     else:
                        #         term_area_time = curr_timestamp - term_area_start_timestamp
                        #         if term_area_time > 0.5:
                        #             terminate = True
                        # #             print('Terminated by the policy!')
                        # else:
                        #     # out of the area
                        #     term_area_start_timestamp = float('inf')

                        if terminate:
                            save_sysid_data()
                            save_gripper_plot()
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            # Interactively label the episode before homing for the next run.
                            prompt_success_fail(episode_id, ee_z)
                            # Robot is homed by the reset at the top of the next episode.
                            print('Episode terminated; restarting.')
                            break

                        # wait for execution
                        precise_wait(t_cycle_end)
                        iter_idx += steps_per_inference

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    save_sysid_data()
                    save_gripper_plot()
                    env.end_episode()
                    if save_video and episode_video_writer is not None:
                        episode_video_writer.close()
                        episode_video_writer = None
                        print(f"  Episode video saved.")
                    if save_video and long_video_writer is not None:
                        long_video_writer.close()
                        long_video_writer = None
                        print(f"  Continuous video saved.")
                    break
                
                print("Stopped.")
                if save_video and episode_video_writer is not None:
                    episode_video_writer.close()
                    episode_video_writer = None
                if save_video and long_video_writer is not None:
                    long_video_writer.close()
                    long_video_writer = None
                    print(f"  Continuous video saved.")



# %%
if __name__ == '__main__':

    main()
