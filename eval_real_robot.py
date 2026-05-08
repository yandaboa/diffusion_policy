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
@click.option('--max_duration', '-md', default=1000, 
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
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints, 
         steps_per_inference, max_duration,
         frequency, save_video, action_noise, contact_threshold,
         collect_sysid):
    # Per-axis Cartesian scale matching simulation DiffIK config
    CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

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
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "415_wrist.json"))
    ]

    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
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
    n_obs_steps = cfg['n_obs_steps']
    n_action_steps = cfg['n_action_steps']
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference: ", steps_per_inference)
    print("n_action_steps: ", n_action_steps)

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
            camera_serial_numbers=['215122255213', '832112070487',
                                  '746112060198'],
            camera_configs=configs,
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
                    # start episode
                    policy.reset()
                    obs_history.clear()
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
                            camera_names = ['front_rgb', 'side_rgb', 'wrist_rgb']
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
                        
                        # action shape: (N, 7) where [:, :6] is Cartesian delta, [:, 6] is gripper
                        raw_arm_action = action[:, :6]  # Raw network output (pre-scale)
                        if action_noise > 0:
                            raw_arm_action = raw_arm_action + np.random.randn(*raw_arm_action.shape) * action_noise
                        gripper_actions = action[:, 6:7]

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

                        raw_actions = np.concatenate([raw_arm_action, gripper_actions], axis=1)  # for last_arm_action obs

                        # Cartesian OSC: scale delta, compute absolute target from observed pose
                        scaled_delta = raw_arm_action * CARTESIAN_SCALE
                        obs_jp = obs['arm_joint_pos'][-1]
                        obs_pos, obs_quat = get_ee_pose(obs_jp)
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
                            env.end_episode()
                            print('Stopped.')
                            break
                        elif key_stroke == ord('r'):
                            # Reset robot and start new trajectory
                            save_sysid_data()
                            sysid_records.clear()
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
                        if time.monotonic() - t_start > max_duration:
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
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            break

                        # wait for execution
                        precise_wait(t_cycle_end)
                        iter_idx += steps_per_inference

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    save_sysid_data()
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
