"""
Eval an RGB DAgger student (rsl_rl ``StudentTeacherVision`` with ResNet18 encoder)
on the real UR5e.

Loads a JIT-exported RGB student (built by ``play.py`` via
``export_vision_student_as_jit``) and feeds it the same proprio + 2-camera RGB
obs the sim wrapper produced at training time:

  proprio:         (1, num_proprio) float32  — concat of history-flattened
                   [prev_actions, arm_joint_pos, end_effector_pose] (history_length=5).
                   joint_pos is the 6 UR5e arm joints only (gripper joints dropped,
                   matching _ProprioCfgArmOnly) → num_proprio = (7+6+6)*5 = 95.
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
import sys
import select
import termios
import tty
import queue
import threading
from collections import deque
from multiprocessing.managers import SharedMemoryManager
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import pathlib
import skvideo.io
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait

import imageio
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# ── Image constants matching rgb_dagger_cfg.py ──────────────────────────────
# 1:1 crop of the 4:3 240×320 render (same as rgb_dagger_cfg IMG_H, IMG_W = 168, 168).
# Real 320×240 capture is squished (not cropped) to this size, matching sim's process_image.
IMG_H, IMG_W = 224, 224

# ── Camera serials must match the order RealEnv enumerates them ─────────────
FRONT_SERIAL = '215122255213'
SIDE_SERIAL  = '832112070487'
WRIST_SERIAL = '746112060198'

# ── Proprio layout (must match _ProprioCfgArmOnly in rgb_dagger_cfg.py) ─────
# The student proprio now uses the 6 UR5e *arm* joints only — the 6 Robotiq
# gripper mimic joints are NOT part of the observation anymore (sim switched
# from ProprioCfg → _ProprioCfgArmOnly). Per-frame layout is therefore
# [prev_action(7), arm_joint_pos(6), ee_pose(6)] = 19, x HISTORY_LEN(5) = 95.
# The gripper joints are still reconstructed below purely for the diagnostic
# finger-position plot, never for the policy input.
HISTORY_LEN     = 1
PREV_ACTION_DIM = 7   # 6 OSC delta + 1 gripper
EE_POSE_DIM     = 6   # 3 pos + 3 axis-angle
NUM_ARM_JOINTS     = 6   # proprio joint_pos = these 6 arm joints only
NUM_GRIPPER_JOINTS = 6   # reconstructed for the finger plot only (not in proprio)

# RTDEInterpolationController normalizes gripper_pos to [0, 1] using the
# hardware-calibrated open/close positions from gripper.get_open/closed_position().
# 0.0 = fully open (master_angle = 0 rad), 1.0 = fully closed (master_angle = π/4 rad).
GRIPPER_POS_OPEN  = 0.0   # calibrated open  (normalized by controller)
GRIPPER_POS_CLOSE = 1.0   # calibrated close (normalized by controller)
GRIPPER_POS_TO_RAD = np.pi / 4 / (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN)
GRIPPER_MIMIC_RATIOS = np.array([
    +1.0,  # col 6 — finger_joint (master)
    +1.0,  # col 7
    -1.0,  # col 8
    +1.0,  # col 9
    -1.0,  # col 10
    -1.0,  # col 11
], dtype=np.float32)


class _KeyReader:
    """Non-blocking keyboard reader via terminal cbreak mode.

    Reads keys directly from stdin rather than relying on cv2/Qt events,
    which break under the multithreaded SharedMemoryManager environment.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._fd: int = sys.stdin.fileno()
        self._old_settings = None
        self._stop_evt = threading.Event()
        self._thread: 'threading.Thread | None' = None

    def start(self) -> '_KeyReader':
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name='_KeyReader')
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        if self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    self._q.put(ord(ch))
            except Exception:
                break

    def get(self) -> int:
        """Return the next pending keycode (int) or -1 if nothing pressed."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return -1


def _process_rgb_image(img: np.ndarray, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Camera RGB frame → (H, W, 3) float32 [0, 1] at policy resolution.

    Uses F.interpolate(bilinear, antialias=True) to match sim's process_image
    exactly. Aspect ratio is not preserved — the image is stretched to (h, w).
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
    """Calibrated FK to wrist_3_link (matching simulation), axis-angle.

    IMPORTANT: canonicalize the quaternion to w>=0 before axis-angle, to match
    IsaacLab's axis_angle_from_quat (which negates q when w<0). Without this, when
    the EE orientation crosses the |angle|=pi singularity (w changes sign — common
    for a downward gripper), quat_to_axis_angle returns the OTHER branch
    (angle in (pi,2pi), opposite axis sign) and the student gets an orientation obs
    that diverges from its sim training convention -> erratic motion.
    """
    n_steps = joint_positions.shape[0]
    ee_poses = np.zeros((n_steps, EE_POSE_DIM), dtype=np.float32)
    for t in range(n_steps):
        pos, quat = get_ee_pose(joint_positions[t])
        quat = np.asarray(quat, dtype=np.float64)
        if quat[0] < 0.0:           # canonicalize w>=0 (matches IsaacLab)
            quat = -quat
        ee_poses[t, :3] = pos
        ee_poses[t, 3:]  = quat_to_axis_angle(quat)
    return ee_poses


def _build_joint_pos(arm_joint_pos: np.ndarray, gripper_pos_raw: float, z_gripper: bool = False) -> np.ndarray:
    """Reconstruct the full 12-dim joint vector (6 arm + 6 gripper mimic joints).

    NOTE: the policy proprio no longer includes the gripper joints (see
    ``_ProprioCfgArmOnly``); this is now used only to feed the diagnostic
    finger-position plot via the ``[6:]`` (gripper) slice. The proprio path
    uses the raw 6 arm joints directly.
    """
    master_angle   = (float(gripper_pos_raw) - GRIPPER_POS_OPEN) * GRIPPER_POS_TO_RAD
    gripper_joints = (GRIPPER_MIMIC_RATIOS * master_angle).astype(np.float32)
    if z_gripper:
        gripper_joints = np.zeros_like(gripper_joints)
    return np.concatenate([arm_joint_pos.astype(np.float32), gripper_joints], axis=0)


def _build_proprio_tensor(history, device) -> torch.Tensor:
    """Stack per-frame history into the policy's proprio input (1, 95):
    [prev_action(7), arm_joint_pos(6), ee_pose(6)] x HISTORY_LEN(5)."""
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
@click.option('--zero_gripper', '-z', is_flag=True, default=False, help="Whether to zero out gripper")
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
def main(zero_gripper, input, output, robot_ip, match_dataset, match_episode,
         init_joints, max_duration, frequency, save_video,
         action_noise, collect_sysid, torch_device):
    CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

    sysid_records = []

    def save_sysid_data():
        if not collect_sysid or len(sysid_records) == 0:
            return
        import torch as _torch
        jp     = np.array([r[0] for r in sysid_records])
        wp_pos  = np.array([r[1] for r in sysid_records])
        wp_quat = np.array([r[2] for r in sysid_records])
        n = len(sysid_records)
        _torch.save({
            "joint_positions":      _torch.tensor(jp,      dtype=_torch.float32),
            "initial_joint_pos":    _torch.tensor(jp[0],   dtype=_torch.float32),
            "waypoint_step_indices": _torch.arange(n,      dtype=_torch.long),
            "waypoint_target_pos":  _torch.tensor(wp_pos,  dtype=_torch.float32),
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
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json")),
    ]

    device = torch.device(
        torch_device if torch.cuda.is_available() and torch_device.startswith('cuda') else 'cpu')
    print(f"Loading RGB policy JIT from {input} on {device}")
    policy = torch.jit.load(input, map_location=device)
    policy.eval()

    meta = _load_jit_metadata(input)
    expected_proprio  = int(meta.get("num_proprio", 0))
    expected_channels = int(meta.get("image_channels", 3))
    expected_groups   = meta.get("vision_groups", "side_rgb,wrist_rgb").split(",")
    # Always use module-level IMG_H/IMG_W — sidecar resolution is stale for
    # finetuned models (reflects pre-finetune training, not the current config).
    policy_h, policy_w = IMG_H, IMG_W
    print(f"[policy resolution] {policy_h}×{policy_w}")

    # The JIT applies one (per-view) encoder per vision group, in this exact
    # order — encoder ``i`` consumes ``images[i]``. We therefore build the image
    # list from ``expected_groups`` (not a hardcoded order) so the right camera
    # always reaches the encoder it was trained with. Each group must map to a
    # camera we know how to capture below.
    KNOWN_RGB_GROUPS = {"side_rgb", "wrist_rgb"}
    unknown = [g for g in expected_groups if g not in KNOWN_RGB_GROUPS]
    if unknown:
        raise ValueError(
            f"JIT declares vision_groups={expected_groups} but this eval script only "
            f"knows how to capture {sorted(KNOWN_RGB_GROUPS)} (unknown: {unknown}). "
            f"Use eval_real_robot_depth.py for depth policies, or add the camera here."
        )
    if expected_channels != 3:
        raise ValueError(
            f"Expected 3-channel RGB images; sidecar reports image_channels={expected_channels}."
        )
    print(f"[vision groups] feeding images in order: {expected_groups}")

    # ── Setup ────────────────────────────────────────────────────────────────
    dt = 1 / frequency
    n_obs_steps = HISTORY_LEN
    print(f"n_obs_steps (matches HISTORY_LEN): {n_obs_steps}")

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                video_capture_resolution=(320, 240),
                obs_image_resolution=(320, 240),
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                enable_depth=False,
                record_raw_video=True,
                rolling_action_buffer=True,
                action_mode='cartesian',
                camera_serial_numbers=[SIDE_SERIAL, WRIST_SERIAL],
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
                "joint_pos":   arm_jp_now.astype(np.float32),  # 6 arm joints only
                "ee_pose":     obs['end_effector_pose'][-1].astype(np.float32),
            }
            warmup_history = deque([warmup_frame] * HISTORY_LEN, maxlen=HISTORY_LEN)
            warmup_proprio = _build_proprio_tensor(warmup_history, device)
            if expected_proprio and warmup_proprio.shape[-1] != expected_proprio:
                raise ValueError(
                    f"proprio dim mismatch: built {warmup_proprio.shape[-1]} but JIT "
                    f"expects {expected_proprio}. Check NUM_ARM_JOINTS and HISTORY_LEN."
                )
            warmup_img = torch.zeros(1, expected_channels, policy_h, policy_w, device=device)
            warmup_images = [warmup_img for _ in expected_groups]
            with torch.no_grad():
                action_mean = policy(warmup_proprio, warmup_images)
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
                print(f"Camera video recording enabled at {video_fps} fps")
                print(f"  Continuous video: {long_video_path}")

            # Terminal key reader — works regardless of cv2/Qt threading issues.
            key_reader = _KeyReader().start()
            print("Key reader active: [S] stop  [R] reset  [G] open gripper")

            actions = []
            gripper_open_steps_remaining = 0
            GRIPPER_OPEN_DURATION      = 5
            STUCK_WINDOW_S             = 2.0
            STUCK_JOINT_THRESHOLD_RAD  = 0.002
            STUCK_GRIPPER_OPEN_STEPS   = int(frequency)
            stuck_buffer = []

            proprio_history: deque = deque(maxlen=HISTORY_LEN)
            last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

            finger_pose_history: list = []
            raw_action_history: list = []

            def save_action_plot():
                if not raw_action_history:
                    return
                import matplotlib.pyplot as plt
                data = np.array(raw_action_history)  # (T, 7)
                arm_labels  = ['dx', 'dy', 'dz', 'dax', 'day', 'daz']
                grip_labels = ['gripper']
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                for j, lbl in enumerate(arm_labels):
                    axes[0].plot(data[:, j], label=lbl)
                axes[0].set_ylabel("Arm delta (raw)")
                axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(data[:, 6], label='gripper', color='tab:orange')
                axes[1].set_ylabel("Gripper (raw)")
                axes[1].set_xlabel("Timestep")
                axes[1].legend(loc="upper right")
                axes[1].grid(True, alpha=0.3)
                fig.suptitle("Raw policy actions over time (real robot)")
                plot_path = os.path.join(output, "raw_actions.png")
                npy_path  = os.path.join(output, "raw_actions.npy")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                np.save(npy_path, data)
                print(f"[INFO] Action plot saved to: {plot_path}")
                print(f"[INFO] Action data saved to: {npy_path}")

            def save_finger_plot():
                if not finger_pose_history:
                    return
                import matplotlib.pyplot as plt
                data = np.array(finger_pose_history)  # (T, num_finger_joints)
                fig, ax = plt.subplots(figsize=(10, 4))
                for j in range(data.shape[1]):
                    ax.plot(data[:, j], label=f"finger_{j}")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Joint position (rad)")
                ax.set_title("Finger joint positions over time (real robot)")
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)
                plot_path = os.path.join(output, "finger_joint_positions.png")
                npy_path  = os.path.join(output, "finger_joint_positions.npy")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                np.save(npy_path, data)
                print(f"[INFO] Finger joint plot saved to: {plot_path}")
                print(f"[INFO] Finger joint data saved to: {npy_path}")

            while True:
                # ========== episode loop =====
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
                    term_area_start_timestamp = float('inf')
                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # ── Get obs ──────────────────────────────────────────
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])

                        # ── Concatenated RGB video ───────────────────────────
                        if save_video:
                            imgs = []
                            for cam in ('side_rgb', 'wrist_rgb'):
                                if cam in obs:
                                    img = obs[cam][-1]
                                    if img.dtype != np.uint8:
                                        img = (img * 255).clip(0, 255).astype(np.uint8)
                                    imgs.append(img)
                            if len(imgs) == 2:
                                frame = np.concatenate(imgs, axis=1)
                                if episode_video_writer is not None:
                                    episode_video_writer.append_data(frame)
                                if long_video_writer is not None:
                                    long_video_writer.append_data(frame)

                        # ── Build RGB policy inputs: (1, 3, H, W) float32 [0,1] ──
                        side_np  = _process_rgb_image(obs['side_rgb'][-1],  h=policy_h, w=policy_w)
                        wrist_np = _process_rgb_image(obs['wrist_rgb'][-1], h=policy_h, w=policy_w)
                        side_t  = torch.from_numpy(
                            side_np.transpose(2, 0, 1)).to(device)[None]      # (1, 3, H, W)
                        wrist_t = torch.from_numpy(
                            wrist_np.transpose(2, 0, 1)).to(device)[None]
                        # Order the encoder inputs by the JIT's vision_groups so
                        # each per-view encoder sees its matching camera.
                        group_to_tensor = {"side_rgb": side_t, "wrist_rgb": wrist_t}
                        policy_images = [group_to_tensor[g] for g in expected_groups]

                        # ── Build proprio ────────────────────────────────────
                        arm_jp_now   = obs['arm_joint_pos'][-1]
                        grip_pos_raw = float(obs['gripper_pos'][-1])
                        finger_pose_history.append(
                            _build_joint_pos(arm_jp_now, grip_pos_raw, z_gripper=zero_gripper)[6:].tolist()
                        )
                        ee_pose_now  = obs['end_effector_pose'][-1].astype(np.float32)
                        proprio_history.append({
                            "prev_action": last_raw_action.copy(),
                            "joint_pos":   arm_jp_now.astype(np.float32),  # 6 arm joints only
                            "ee_pose":     ee_pose_now,
                        })
                        proprio_tensor = _build_proprio_tensor(proprio_history, device)

                        # ── Run inference ────────────────────────────────────
                        with torch.no_grad():
                            action_mean = policy(proprio_tensor, policy_images).cpu().numpy()
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
                                range_per_joint = jps.max(axis=0) - jps.min(axis=0)
                                if np.max(range_per_joint) < STUCK_JOINT_THRESHOLD_RAD:
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
                        raw_action_history.append(last_raw_action.copy())

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

                        # ── Schedule action (latency-compensated) ────────────
                        action_timestamps = np.arange(1, dtype=np.float64) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # Inference exceeded time budget — snap to next grid step.
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                        else:
                            action_timestamps = action_timestamps[is_new]

                        actions.append(target_actions)
                        env.exec_actions(
                            actions=target_actions,
                            timestamps=action_timestamps,
                            obs_actions=raw_actions,
                        )

                        # ── Visualize: raw capture + policy input ────────────
                        episode_id = env.replay_buffer.n_episodes
                        status_text = f'Ep:{episode_id}  t:{time.monotonic() - t_start:.1f}s'

                        def _to_uint8_bgr(arr: np.ndarray) -> np.ndarray:
                            if arr.dtype != np.uint8:
                                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                            return arr[..., ::-1]  # RGB→BGR for cv2

                        raw_panels, pol_panels = [], []
                        for cam_key, cam_t in (('side_rgb', side_t), ('wrist_rgb', wrist_t)):
                            if cam_key in obs:
                                raw = _to_uint8_bgr(obs[cam_key][-1])  # (240, 320, 3) BGR
                                raw_panels.append(raw)
                                # policy input: (1,3,H,W) float [0,1] → (H,W,3) uint8 BGR
                                pol_np = cam_t[0].permute(1, 2, 0).cpu().numpy()
                                pol_np = (pol_np * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
                                # upscale to 240×320 for side-by-side readability
                                pol_up = cv2.resize(pol_np, (320, 240), interpolation=cv2.INTER_NEAREST)
                                pol_panels.append(pol_up)

                        if raw_panels:
                            raw_row = np.concatenate(raw_panels, axis=1)  # (240, 320*N, 3)
                            pol_row = np.concatenate(pol_panels, axis=1) if pol_panels else np.zeros_like(raw_row)
                            div = np.full((4, raw_row.shape[1], 3), 128, dtype=np.uint8)
                            vis = np.concatenate([raw_row, div, pol_row], axis=0)
                            cv2.putText(vis, f'RAW  {status_text}', (8, 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(vis, f'POLICY INPUT ({IMG_H}x{IMG_W})', (8, 244 + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                            cv2.imshow('Policy Control', vis)
                        cv2.waitKey(1)  # refresh display only — key detection via key_reader

                        key_stroke = key_reader.get()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening gripper for {GRIPPER_OPEN_DURATION} steps")
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
                                print(f"  Episode video saved.")
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
                            term_area_start_timestamp = float('inf')
                            print('Robot reset complete! Starting new trajectory.')
                            continue

                        if time.monotonic() - t_start > max_duration:
                            print('Terminated by the timeout!')
                            save_sysid_data()
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            print('Auto-resetting robot to initial position...')
                            env.robot.reset_to_initial_position()
                            time.sleep(5.0)
                            print('Reset complete, restarting episode.')
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    save_sysid_data()
                    save_finger_plot()
                    save_action_plot()
                    env.end_episode()
                    key_reader.stop()
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
                save_finger_plot()
                save_action_plot()
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
