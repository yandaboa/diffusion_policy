"""
Eval an RGB DAgger student (rsl_rl ``StudentTeacherVision`` with ResNet18 encoder)
on the real UR5e.

Loads a JIT-exported RGB student (built by ``play.py`` via
``export_vision_student_as_jit``) and feeds it the same proprio + 2-camera RGB
obs the sim wrapper produced at training time:

  proprio:         (1, num_proprio) float32  — concat of history-flattened
                   [prev_actions, joint_pos, end_effector_pose] (history_length=5)
  side_rgb, wrist_rgb: (1, 3, 224, 224) float32 in [0, 1]

No depth processing or DA3 required — raw camera RGB is resized and passed directly.

Usage:
(robodiff_real)$ python eval_real_robot_rgb.py -i <rgb_policy_jit> -o <save_dir> --robot_ip <ip>

================ Human in control ==============
Move the SpaceMouse to position the robot. Press "C" to hand control to the
policy, "Q" to exit.

================ Policy in control ==============
Press "S" to stop, "R" to reset robot to initial joints, "G" to force the
gripper open for a few steps (also auto-triggered after 2 s of no motion).
"""

# %%
import time
from collections import deque
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import json
import pathlib
import skvideo.io
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait

import imageio
import torch.nn.functional as F
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# ── Image constants matching rgb_dagger_cfg.py ──────────────────────────────
IMG_H, IMG_W = 224, 224

# ── Camera serials must match the order RealEnv enumerates them ─────────────
FRONT_SERIAL = '215122255213'
SIDE_SERIAL  = '832112070487'
WRIST_SERIAL = '746112060198'

# ── Proprio layout (must match DepthDAggerObservationsCfg.ProprioCfg) ───────
HISTORY_LEN     = 5
PREV_ACTION_DIM = 7   # 6 OSC delta + 1 gripper
EE_POSE_DIM     = 6   # 3 pos + 3 axis-angle
NUM_ARM_JOINTS     = 6
NUM_GRIPPER_JOINTS = 6
NUM_JOINTS = NUM_ARM_JOINTS + NUM_GRIPPER_JOINTS  # 12

GRIPPER_POS_TO_RAD = np.pi / 4 / 255.0
GRIPPER_MIMIC_RATIOS = np.array([+1.0, +1.0, -1.0, +1.0, -1.0, -1.0], dtype=np.float32)


def _process_rgb_image(img: np.ndarray, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Camera RGB frame → (H, W, 3) float32 [0, 1] at policy resolution.

    Matches sim's process_image exactly:
      img / 255 → clamp [0,1] → F.interpolate(bilinear, antialias=True)
    The antialias flag applies a Gaussian pre-filter before the bilinear
    downsample (640×480 → 224×224), which is what the student saw at training
    time. cv2.INTER_LINEAR skips this and produces aliased high-freq content
    the ResNet18 encoder never encountered.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if img.shape[:2] != (h, w):
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H_in, W_in)
        t = F.interpolate(t, size=(h, w), mode='bilinear', antialias=True)
        img = t.squeeze(0).permute(1, 2, 0).numpy()                # (H, W, 3)
    return img


def compute_calibrated_ee_pose(joint_positions: np.ndarray) -> np.ndarray:
    """Calibrated FK to wrist_3_link (matching simulation)."""
    n_steps = joint_positions.shape[0]
    ee_poses = np.zeros((n_steps, EE_POSE_DIM), dtype=np.float32)
    for t in range(n_steps):
        pos, quat = get_ee_pose(joint_positions[t])
        ee_poses[t, :3] = pos
        ee_poses[t, 3:]  = quat_to_axis_angle(quat)
    return ee_poses


def _build_joint_pos(arm_joint_pos: np.ndarray, gripper_pos_raw: float) -> np.ndarray:
    master_angle  = float(gripper_pos_raw) * GRIPPER_POS_TO_RAD
    gripper_joints = (GRIPPER_MIMIC_RATIOS * master_angle).astype(np.float32)
    return np.concatenate([arm_joint_pos.astype(np.float32), gripper_joints], axis=0)


def _build_proprio_tensor(history, device) -> torch.Tensor:
    """Stack per-frame history into the policy's proprio input (1, 125)."""
    if len(history) == 0:
        raise ValueError("Empty proprio history")
    while len(history) < HISTORY_LEN:
        history.appendleft(history[0])
    prev_actions = np.concatenate([h["prev_action"] for h in history], axis=0)
    joint_pos    = np.concatenate([h["joint_pos"]    for h in history], axis=0)
    ee_pose      = np.concatenate([h["ee_pose"]      for h in history], axis=0)
    flat = np.concatenate([prev_actions, joint_pos, ee_pose], axis=0).astype(np.float32)
    return torch.from_numpy(flat).unsqueeze(0).to(device)


def _load_jit_metadata(jit_path: str) -> dict:
    """Read sidecar metadata (<stem>_meta.txt) written by the exporter."""
    meta_path = pathlib.Path(jit_path).with_name(
        pathlib.Path(jit_path).stem + "_meta.txt")
    meta = {}
    if not meta_path.exists():
        print(f"[WARN] No sidecar metadata at {meta_path}; will skip shape validation.")
        return meta
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k] = v
    print(f"[loaded JIT metadata] {meta}")
    return meta


@click.command()
@click.option('--input', '-i', required=True,
              help='Path to JIT-exported RGB student (rgb_policy.pt).')
@click.option('--output', '-o', required=True,
              help='Directory to save recording.')
@click.option('--robot_ip', '-ri', required=True,
              help="UR5's IP address e.g. 192.168.1.10")
@click.option('--match_dataset', '-m', default=None,
              help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int,
              help='Match specific episode from the match dataset')
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--max_duration', '-md', default=1000,
              help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float,
              help="Control frequency in Hz.")
@click.option('--save_video', is_flag=True, default=False,
              help='Save video of concatenated camera views.')
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise added to raw arm actions (pre-scale).')
@click.option('--collect_sysid', default=None, type=str,
              help='Save on-policy sysid data to .pt file (joint traj + OSC targets)')
@click.option('--torch_device', default='cuda', type=str,
              help='Torch device for JIT inference.')
def main(input, output, robot_ip, match_dataset, match_episode,
         init_joints, max_duration, frequency, save_video,
         action_noise, collect_sysid, torch_device):
    CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

    sysid_records = []

    def save_sysid_data():
        if not collect_sysid or len(sysid_records) == 0:
            return
        import torch as _torch
        jp    = np.array([r[0] for r in sysid_records])
        wp_pos  = np.array([r[1] for r in sysid_records])
        wp_quat = np.array([r[2] for r in sysid_records])
        n = len(sysid_records)
        _torch.save({
            "joint_positions":     _torch.tensor(jp, dtype=_torch.float32),
            "initial_joint_pos":   _torch.tensor(jp[0], dtype=_torch.float32),
            "waypoint_step_indices": _torch.arange(n, dtype=_torch.long),
            "waypoint_target_pos": _torch.tensor(wp_pos,  dtype=_torch.float32),
            "waypoint_target_quat": _torch.tensor(wp_quat, dtype=_torch.float32),
            "dt": dt,
        }, collect_sysid)
        print(f"\nSaved sysid data ({n} steps at {frequency} Hz) to: {collect_sysid}")

    # ── match_dataset (kept for parity with eval_real_robot.py) ─────────────
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        for vid_dir in match_dir.joinpath('videos').glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir / f'{match_camera_idx}.mp4'
            if match_video_path.exists():
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # ── Load RGB policy + validate metadata ─────────────────────────────────
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json")),
    ]

    device = torch.device(
        torch_device if torch.cuda.is_available() and torch_device.startswith('cuda') else 'cpu')
    print(f"Loading RGB policy JIT from {input} on {device}")
    policy = torch.jit.load(input, map_location=device)
    policy.eval()

    meta = _load_jit_metadata(input)
    expected_proprio   = int(meta.get("num_proprio", 0))
    expected_h         = int(meta.get("image_h", IMG_H))
    expected_w         = int(meta.get("image_w", IMG_W))
    expected_channels  = int(meta.get("image_channels", 3))
    expected_groups    = meta.get("vision_groups", "side_rgb,wrist_rgb").split(",")

    if expected_groups != ["side_rgb", "wrist_rgb"]:
        raise ValueError(
            f"This eval script expects vision_groups=['side_rgb', 'wrist_rgb']; "
            f"the JIT was trained with {expected_groups}. "
            f"Use eval_real_robot_depth.py for depth policies."
        )
    if expected_channels != 3:
        raise ValueError(
            f"Expected 3-channel RGB images; sidecar reports image_channels={expected_channels}."
        )
    if (expected_h, expected_w) != (IMG_H, IMG_W):
        raise ValueError(
            f"JIT expects {expected_h}×{expected_w} images, but this script resizes to "
            f"{IMG_H}×{IMG_W}. Update IMG_H/IMG_W constants to match."
        )

    # ── Setup ────────────────────────────────────────────────────────────────
    dt = 1 / frequency
    n_obs_steps = HISTORY_LEN

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=(640, 480),
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                enable_depth=False,
                record_raw_video=True,
                rolling_action_buffer=True,
                action_mode='cartesian',
                camera_serial_numbers=[FRONT_SERIAL, SIDE_SERIAL, WRIST_SERIAL],
                camera_configs=configs,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)

            print("Waiting for realsense")
            time.sleep(5.0)

            # ── Warmup policy inference ──────────────────────────────────────
            print("Warming up policy inference")
            obs = env.get_obs()
            obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])
            arm_jp_now = obs['arm_joint_pos'][-1]
            grip_now   = float(obs['gripper_pos'][-1])
            warmup_frame = {
                "prev_action": np.zeros(PREV_ACTION_DIM, dtype=np.float32),
                "joint_pos":   _build_joint_pos(arm_jp_now, grip_now),
                "ee_pose":     obs['end_effector_pose'][-1].astype(np.float32),
            }
            warmup_history = deque([warmup_frame] * HISTORY_LEN, maxlen=HISTORY_LEN)
            warmup_proprio = _build_proprio_tensor(warmup_history, device)
            if expected_proprio and warmup_proprio.shape[-1] != expected_proprio:
                raise ValueError(
                    f"proprio dim mismatch: built {warmup_proprio.shape[-1]} but JIT "
                    f"expects {expected_proprio}. Check NUM_JOINTS and HISTORY_LEN."
                )
            warmup_img = torch.zeros(1, expected_channels, IMG_H, IMG_W, device=device)
            with torch.no_grad():
                action_mean = policy(warmup_proprio, [warmup_img, warmup_img])
                print(f"Warmup OK; action shape={tuple(action_mean.shape)}")
                del action_mean

            print('Ready!')
            time.sleep(1.0)

            # ── Video writers ────────────────────────────────────────────────
            video_fps = int(frequency)
            episode_video_writer = None
            long_video_writer    = None
            if save_video:
                long_video_path = pathlib.Path(output) / 'policy_cameras_full.mp4'
                long_video_writer = imageio.get_writer(
                    str(long_video_path), fps=video_fps, codec='libx264',
                    output_params=['-crf', '21', '-preset', 'fast'])
                print(f"Camera video: {long_video_path}")

            actions = []
            gripper_open_steps_remaining = 0
            GRIPPER_OPEN_DURATION      = 5
            STUCK_WINDOW_S             = 2.0
            STUCK_JOINT_THRESHOLD_RAD  = 0.002
            STUCK_GRIPPER_OPEN_STEPS   = int(frequency)
            stuck_buffer = []

            proprio_history: deque = deque(maxlen=HISTORY_LEN)
            last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

            while True:
                # ===== episode loop =====
                try:
                    proprio_history.clear()
                    last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)
                    start_delay   = 1.0
                    eval_t_start  = time.time() + start_delay
                    t_start       = time.monotonic() + start_delay
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
                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # ── Get obs ──────────────────────────────────────────
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])

                        # ── Concatenated RGB video ───────────────────────────
                        if save_video:
                            imgs = []
                            for cam in ('front_rgb', 'side_rgb', 'wrist_rgb'):
                                if cam in obs:
                                    img = obs[cam][-1]
                                    if img.dtype != np.uint8:
                                        img = (img * 255).clip(0, 255).astype(np.uint8)
                                    imgs.append(img)
                            if len(imgs) == 3:
                                frame = np.concatenate(imgs, axis=1)
                                if episode_video_writer is not None:
                                    episode_video_writer.append_data(frame)
                                if long_video_writer is not None:
                                    long_video_writer.append_data(frame)

                        # ── Build RGB policy inputs: (1, 3, H, W) float32 [0,1] ──
                        side_np  = _process_rgb_image(obs['side_rgb'][-1])   # (H, W, 3)
                        wrist_np = _process_rgb_image(obs['wrist_rgb'][-1])
                        side_t   = torch.from_numpy(
                            side_np.transpose(2, 0, 1)).to(device)[None]     # (1, 3, H, W)
                        wrist_t  = torch.from_numpy(
                            wrist_np.transpose(2, 0, 1)).to(device)[None]

                        # ── Build proprio ────────────────────────────────────
                        arm_jp_now    = obs['arm_joint_pos'][-1]
                        grip_pos_raw  = float(obs['gripper_pos'][-1])
                        ee_pose_now   = obs['end_effector_pose'][-1].astype(np.float32)
                        proprio_history.append({
                            "prev_action": last_raw_action.copy(),
                            "joint_pos":   _build_joint_pos(arm_jp_now, grip_pos_raw),
                            "ee_pose":     ee_pose_now,
                        })
                        proprio_tensor = _build_proprio_tensor(proprio_history, device)

                        # ── Run inference ────────────────────────────────────
                        with torch.no_grad():
                            action_mean = policy(proprio_tensor, [side_t, wrist_t]).cpu().numpy()
                        raw_action = action_mean[0]  # (7,)
                        if action_noise > 0:
                            raw_action[:6] = raw_action[:6] + np.random.randn(6) * action_noise

                        # ── Stuck detection ──────────────────────────────────
                        if gripper_open_steps_remaining == 0:
                            t_now = time.monotonic()
                            stuck_buffer.append((t_now, arm_jp_now.copy()))
                            while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                stuck_buffer.pop(0)
                            if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                jps = np.array([b[1] for b in stuck_buffer])
                                if np.max(jps.max(axis=0) - jps.min(axis=0)) < STUCK_JOINT_THRESHOLD_RAD:
                                    gripper_open_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                    stuck_buffer.clear()
                                    print("[Stuck detection] No movement for 2 s, opening gripper")

                        # ── Gripper open macro ───────────────────────────────
                        gripper_action = raw_action[6:7].copy()
                        if gripper_open_steps_remaining > 0:
                            gripper_action = np.array([1.0], dtype=np.float32)
                            gripper_open_steps_remaining -= 1
                            if gripper_open_steps_remaining == 0:
                                print("[Gripper macro] done, returning to policy control")
                        last_raw_action = np.concatenate(
                            [raw_action[:6], gripper_action]).astype(np.float32)

                        # ── Cartesian OSC: scale delta → absolute target ──────
                        scaled_delta = raw_action[:6] * CARTESIAN_SCALE
                        obs_pos, obs_quat = get_ee_pose(arm_jp_now)
                        tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled_delta)
                        tgt_aa = quat_to_axis_angle(tgt_quat)
                        abs_target  = np.concatenate([tgt_pos, tgt_aa])[None]       # (1, 6)
                        target_actions = np.concatenate([abs_target, gripper_action[None]], axis=1)
                        raw_actions    = np.concatenate(
                            [raw_action[:6][None], gripper_action[None]], axis=1)

                        if collect_sysid:
                            sysid_records.append((arm_jp_now.copy(), tgt_pos.copy(), tgt_quat.copy()))

                        # ── Schedule action ──────────────────────────────────
                        action_timestamp = float(obs_timestamps[-1] + dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        if action_timestamp <= curr_time + action_exec_latency:
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + next_step_idx * dt

                        actions.append(target_actions)
                        env.exec_actions(
                            actions=target_actions,
                            timestamps=np.array([action_timestamp]),
                            obs_actions=raw_actions,
                        )

                        # ── Visualize (side camera) ──────────────────────────
                        episode_id = env.replay_buffer.n_episodes
                        if 'side_rgb' in obs:
                            vis_img = obs['side_rgb'][-1]
                            if vis_img.dtype != np.uint8:
                                vis_img = (np.clip(vis_img, 0.0, 1.0) * 255).astype(np.uint8)
                            text = f'Episode: {episode_id}, Time: {time.monotonic() - t_start:.1f}'
                            cv2.putText(vis_img, text, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.imshow('Policy Control', vis_img[..., ::-1])

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening for {GRIPPER_OPEN_DURATION} steps")
                        elif key_stroke == ord('s'):
                            save_sysid_data()
                            env.end_episode()
                            print('Stopped.')
                            break
                        elif key_stroke == ord('r'):
                            save_sysid_data()
                            sysid_records.clear()
                            stuck_buffer.clear()
                            print('Resetting robot for new trajectory...')
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print("  Episode video saved.")
                            proprio_history.clear()
                            last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)
                            env.robot.reset_to_initial_position()
                            time.sleep(5.0)
                            start_delay  = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start      = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            precise_wait(eval_t_start, time_func=time.time)
                            iter_idx = 0
                            print('Robot reset complete! Starting new trajectory.')
                            continue

                        if time.monotonic() - t_start > max_duration:
                            print('Terminated by the timeout!')
                            save_sysid_data()
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    save_sysid_data()
                    env.end_episode()
                    if save_video and episode_video_writer is not None:
                        episode_video_writer.close()
                        episode_video_writer = None
                    if save_video and long_video_writer is not None:
                        long_video_writer.close()
                        long_video_writer = None
                    break

                print("Stopped.")
                if save_video and episode_video_writer is not None:
                    episode_video_writer.close()
                    episode_video_writer = None
                if save_video and long_video_writer is not None:
                    long_video_writer.close()
                    long_video_writer = None


# %%
if __name__ == '__main__':
    main()
