"""
Eval a depth-DAgger student (rsl_rl ``StudentTeacherVision``) on the real UR5e.

Loads a JIT-exported depth student (built by ``play.py`` via
``export_vision_student_as_jit``) and feeds it the same proprio + 2-camera depth
obs the sim wrapper produced at training time:

  proprio:   (1, num_proprio) float32  — concat of history-flattened
             [prev_actions, joint_pos, end_effector_pose] (history_length=5)
  side_depth, wrist_depth: (1, 1, 224, 224) float32 in [0,1], clipped at
             ``DEPTH_CLIP = (0.01, 2.0)`` m, with no-return pixels mapped to d_max.
             RealSense u16 + DA3METRIC fusion mirrors ``demo_real_robot.py``.

Usage:
(robodiff_real)$ python eval_real_robot_depth.py -i <depth_policy_jit> -o <save_dir> --robot_ip <ip>

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
from diffusion_policy.real_world.da3_depth_client import DA3DepthClient, DA3DepthBackground

# Add imageio import for video saving
import imageio
from scipy.spatial.transform import Rotation as R

# Calibrated FK matching simulation (wrist_3_link in REP-103 base_link frame)
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# ── Depth constants matching depth_dagger_cfg.py ───────────────────────────
DEPTH_CLIP = (0.01, 2.0)            # metres, must match sim's DEPTH_CLIP
DEPTH_IMG_H, DEPTH_IMG_W = 224, 224 # must match sim's IMG_H, IMG_W

# ── Camera serials must match the order RealEnv enumerates them ─────────────
# (camera_idx 0 = front, 1 = side, 2 = wrist in real_env.get_obs).
FRONT_SERIAL = '215122255213'
SIDE_SERIAL  = '832112070487'
WRIST_SERIAL = '746112060198'

# ── Proprio layout (must match DepthDAggerObservationsCfg.ProprioCfg) ───────
# Per-frame: prev_actions (7) + joint_pos (12) + end_effector_pose (6) = 25
# History length 5, terms concatenated: 5*7 + 5*12 + 5*6 = 125 dims total.
HISTORY_LEN = 5
PREV_ACTION_DIM = 7  # 6 OSC delta + 1 gripper
EE_POSE_DIM = 6      # 3 pos + 3 axis-angle
# Robotiq 2F85 has 6 internal joints driven by the master finger_joint via
# mimic constraints. Combined with 6 UR5e arm joints → 12-dim joint_pos.
NUM_ARM_JOINTS = 6
NUM_GRIPPER_JOINTS = 6
NUM_JOINTS = NUM_ARM_JOINTS + NUM_GRIPPER_JOINTS  # 12

# RTDEInterpolationController normalizes gripper_pos to [0, 1] using the
# hardware-calibrated open/close positions from gripper.get_open/closed_position().
# 0.0 = fully open (master_angle = 0 rad), 1.0 = fully closed (master_angle = π/4 rad).
GRIPPER_POS_OPEN  = 0.0   # calibrated open  (normalized by controller)
GRIPPER_POS_CLOSE = 1.0   # calibrated close (normalized by controller)
GRIPPER_POS_TO_RAD = np.pi / 4 / (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN)

# Mimic-ratio pattern for the 6 gripper joints w.r.t. the finger_joint master,
# in the articulation order Isaac Lab returns ``robot.data.joint_pos``.
# Empirically extracted from a closed-gripper reset state in
#   /mnt/storage/lti/UWLab/reset_states_dataset_small/Resets/Peg/
#       resets_ObjectAnywhereEEAnywhere.pt
# (12-dim joint_position; columns 6..11 = +0.78, +0.78, -0.78, +0.78, -0.78, -0.78
# at master angle ≈ +π/4). Per-joint identification by Robotiq 2F-85 URDF
# convention is NOT needed for the proprio vector — only the per-column sign
# w.r.t. the master matters.
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

    Usage:
        reader = _KeyReader()
        reader.start()
        ...
        key = reader.get()   # returns ord(char) or -1 if nothing pending
        ...
        reader.stop()
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._fd: int = sys.stdin.fileno()
        self._old_settings = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> '_KeyReader':
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)   # char-by-char input; keeps Ctrl+C working
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='_KeyReader')
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


def _process_realsense_depth(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    """Raw u16 RealSense depth → policy-input float32 [0,1] @ 224×224.

    Mirrors sim's ``process_image`` for ``data_type='distance_to_camera'``:
      1. Convert u16 → metres via ``depth_scale`` (m/unit from sensor).
      2. Map zero-pixels (no return) → ``d_max`` so they read as far range,
         matching the sim path which sets nan/inf → d_max via ``nan_to_num``.
      3. Clip to ``DEPTH_CLIP`` and normalise to [0, 1].
      4. Bilinear resize to 224×224 (sim uses bilinear+antialias on GPU; cv2
         INTER_LINEAR is the closest CPU equivalent).
    """
    d_min, d_max = DEPTH_CLIP
    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_m[depth_m == 0.0] = d_max
    np.clip(depth_m, d_min, d_max, out=depth_m)
    depth_norm = (depth_m - d_min) / (d_max - d_min)
    return cv2.resize(depth_norm, (DEPTH_IMG_W, DEPTH_IMG_H), interpolation=cv2.INTER_LINEAR)


def _process_metric_depth(depth_m: np.ndarray) -> np.ndarray:
    """DA3-fused metric depth (float32 metres) → policy-input float32 [0,1] @ 224×224.

    Same clip+normalise+resize as ``_process_realsense_depth`` but the input
    is already metric float (no u16 → m conversion). DA3 fills holes during
    fusion, so zero-as-no-return handling is unnecessary; we still nan-guard
    in case fusion returned nan/inf.
    """
    d_min, d_max = DEPTH_CLIP
    depth_m = np.nan_to_num(depth_m, nan=d_max, posinf=d_max, neginf=d_max)
    depth_m = np.clip(depth_m, d_min, d_max)
    depth_norm = (depth_m - d_min) / (d_max - d_min)
    return cv2.resize(depth_norm, (DEPTH_IMG_W, DEPTH_IMG_H), interpolation=cv2.INTER_LINEAR)


def _depth_to_bgr(depth_norm: np.ndarray) -> np.ndarray:
    """float32 [0,1] → BGR uint8 via TURBO colormap (matches demo_real_robot.py)."""
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _metric_to_bgr(depth_m: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """float32 metric depth (metres) → BGR uint8 via TURBO, resized to out_wh (W, H)."""
    d_min, d_max = DEPTH_CLIP
    depth = np.nan_to_num(depth_m, nan=d_max, posinf=d_max, neginf=d_max).astype(np.float32)
    depth = np.clip(depth, d_min, d_max)
    norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    if norm.shape[:2][::-1] != out_wh:
        norm = cv2.resize(norm, out_wh, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _ee_to_w2c(pos: np.ndarray, quat_wxyz: np.ndarray,
               cam_in_ee: np.ndarray) -> np.ndarray:
    """Compute world-to-camera 4×4 from FK output + camera-in-wrist offset.

    Args:
        pos:        (3,) wrist_3_link position in base frame.
        quat_wxyz:  (4,) quaternion [w, x, y, z] from get_ee_pose().
        cam_in_ee:  (4, 4) camera-to-wrist_3_link transform (calibration file).

    Returns: (4, 4) world-to-camera matrix, float64.
    """
    w, x, y, z = quat_wxyz
    R_ee = R.from_quat([x, y, z, w]).as_matrix()  # scipy expects xyzw
    T_ee = np.eye(4, dtype=np.float64)
    T_ee[:3, :3] = R_ee
    T_ee[:3,  3] = pos
    T_cam = T_ee @ cam_in_ee          # camera-in-world
    return np.linalg.inv(T_cam)       # world-to-camera


def compute_calibrated_ee_pose(joint_positions: np.ndarray) -> np.ndarray:
    """Compute EE pose using calibrated FK to wrist_3_link (matching simulation).

    Args:
        joint_positions: (n, 6) joint angles in radians.
    Returns:
        ee_poses: (n, 6) [x, y, z, rx, ry, rz] axis-angle.
    """
    n_steps = joint_positions.shape[0]
    ee_poses = np.zeros((n_steps, EE_POSE_DIM), dtype=np.float32)
    for t in range(n_steps):
        pos, quat = get_ee_pose(joint_positions[t])
        axis_angle = quat_to_axis_angle(quat)
        ee_poses[t, :3] = pos
        ee_poses[t, 3:] = axis_angle
    return ee_poses


def _build_joint_pos(arm_joint_pos: np.ndarray, gripper_pos_raw: float) -> np.ndarray:
    """Reconstruct the 12-dim sim joint_pos from real-robot scalars.

    Args:
        arm_joint_pos: (6,) UR5e joint angles (rad).
        gripper_pos_raw: normalized gripper position in [0, 1] (0=open, 1=closed),
            as returned by RTDEInterpolationController using calibrated open/close positions.
    Returns: (12,) float32 in the order Isaac Lab returns
        ``asset.data.joint_pos`` for EXPLICIT_UR5E_ROBOTIQ_2F85
        (arm joints first, then 6 gripper joints driven by mimic).
    """
    master_angle = (float(gripper_pos_raw) - GRIPPER_POS_OPEN) * GRIPPER_POS_TO_RAD
    gripper_joints = (GRIPPER_MIMIC_RATIOS * master_angle).astype(np.float32)
    return np.concatenate([arm_joint_pos.astype(np.float32), gripper_joints], axis=0)


def _build_proprio_tensor(history, device) -> torch.Tensor:
    """Stack the per-frame history into the policy's proprio input.

    Args:
        history: deque of dicts {prev_action: (7,), joint_pos: (12,), ee_pose: (6,)},
            oldest-first. Pads short histories by repeating the earliest frame
            (matches Isaac Lab's CircularBuffer, which back-fills with the first
            observation seen at startup).

    Returns: (1, 5*7 + 5*12 + 5*6) = (1, 125) tensor on ``device``.
    Layout (matches Isaac Lab ObservationManager with concatenate_terms=True
    and flatten_history_dim=True): all 5 prev_action frames flat, then all 5
    joint_pos frames flat, then all 5 ee_pose frames flat.
    """
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
    """Read sidecar metadata (depth_policy_meta.txt) written by the exporter."""
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
              help='Path to JIT-exported depth student (depth_policy.pt).')
@click.option('--output', '-o', required=True,
              help='Directory to save recording.')
@click.option('--robot_ip', '-ri', required=True,
              help="UR5's IP address e.g. 192.168.1.10")
@click.option('--match_dataset', '-m', default=None,
              help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int,
              help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int,
              help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--max_duration', '-md', default=1000,
              help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float,
              help="Control frequency in Hz.")
@click.option('--save_video', is_flag=True, default=False,
              help='Save video of concatenated camera views + depth panel.')
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise added to raw arm actions (pre-scale).')
@click.option('--collect_sysid', default=None, type=str,
              help='Save on-policy sysid data to .pt file (joint traj + OSC targets)')
@click.option('--use_da3_fusion/--no_da3_fusion', default=True,
              help='Use DA3 metric depth fusion with RealSense (better at hole-y/specular pixels).')
@click.option('--da3_only', is_flag=True, default=False,
              help='Feed raw DA3 metric depth to the policy, skipping RealSense entirely (implies DA3 is used).')
@click.option('--use_nested', is_flag=True, default=False,
              help='Use DA3NESTED-GIANT-LARGE (multi-view history=2) instead of DA3METRIC-LARGE.')
@click.option('--da3_process_res', default=378, type=int, show_default=True,
              help='DA3NESTED processing resolution (252=~78ms, 378=~146ms, 504=~241ms on RTX4090).')
@click.option('--wrist_cam_extrinsic', default=None, type=str,
              help='Path to (4,4) float64 .npy: camera-to-wrist_3_link transform (for pose conditioning).')
@click.option('--side_cam_extrinsic', default=None, type=str,
              help='Path to (4,4) float64 .npy: world-to-side-camera transform (for pose conditioning).')
@click.option('--torch_device', default='cuda', type=str,
              help='Torch device for JIT inference.')
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints, max_duration,
         frequency, save_video, action_noise,
         collect_sysid, use_da3_fusion, da3_only,
         use_nested, da3_process_res, wrist_cam_extrinsic, side_cam_extrinsic,
         torch_device):
    # Per-axis Cartesian scale matching simulation DiffIK config.
    # Identical to eval_real_robot.py — sim's RelCartesianOSCEvalAction scales.
    CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")
    if da3_only:
        use_da3_fusion = True
        print("[da3_only] Bypassing RealSense — feeding raw DA3 metric depth to policy.")
    if use_nested:
        use_da3_fusion = True
        print("[use_nested] DA3NESTED-GIANT-LARGE enabled (multi-view history=2).")

    # Select DA3 model and load optional pose-conditioning calibration matrices
    da3_model_id = ('depth-anything/DA3NESTED-GIANT-LARGE' if use_nested
                    else 'depth-anything/DA3METRIC-LARGE')

    wrist_cam_T: 'np.ndarray | None' = None   # camera-in-wrist_3_link (4×4)
    side_cam_w2c: 'np.ndarray | None' = None  # world-to-side-cam (4×4)
    if use_nested and wrist_cam_extrinsic:
        wrist_cam_T = np.load(wrist_cam_extrinsic).astype(np.float64)
        assert wrist_cam_T.shape == (4, 4), "--wrist_cam_extrinsic must be (4,4)"
        print(f"[pose] Wrist cam-in-EE transform loaded from {wrist_cam_extrinsic}")
    if use_nested and side_cam_extrinsic:
        side_cam_w2c = np.load(side_cam_extrinsic).astype(np.float64)
        assert side_cam_w2c.shape == (4, 4), "--side_cam_extrinsic must be (4,4)"
        print(f"[pose] Side world-to-cam transform loaded from {side_cam_extrinsic}")

    use_pose = use_nested and (wrist_cam_T is not None) and (side_cam_w2c is not None)
    if use_nested and not use_pose:
        print("[pose] No extrinsic files provided — running NESTED without pose conditioning.")

    sysid_records = []  # list of (joint_pos, target_pos, target_quat)

    def save_sysid_data():
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

    # load match_dataset (kept for parity with eval_real_robot.py)
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # ── Load depth policy + metadata ───────────────────────────────────────
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json")),
    ]

    device = torch.device(torch_device if torch.cuda.is_available() and torch_device.startswith('cuda') else 'cpu')
    print(f"Loading depth policy JIT from {input} on {device}")
    policy = torch.jit.load(input, map_location=device)
    policy.eval()
    meta = _load_jit_metadata(input)
    expected_proprio = int(meta.get("num_proprio", 0))
    expected_h = int(meta.get("image_h", DEPTH_IMG_H))
    expected_w = int(meta.get("image_w", DEPTH_IMG_W))
    expected_groups = meta.get("vision_groups", "side_depth,wrist_depth").split(",")
    if expected_groups != ["side_depth", "wrist_depth"]:
        raise ValueError(
            f"This eval script is wired for vision_groups=['side_depth', 'wrist_depth']; "
            f"JIT was trained with {expected_groups}. Update the camera plumbing if "
            f"the policy expects a different camera set."
        )
    if (expected_h, expected_w) != (DEPTH_IMG_H, DEPTH_IMG_W):
        raise ValueError(
            f"JIT expects {expected_h}x{expected_w} depth, but this script always "
            f"resizes to {DEPTH_IMG_H}x{DEPTH_IMG_W}. Update DEPTH_IMG_H/W to match."
        )

    # ── setup experiments ──────────────────────────────────────────────────
    dt = 1 / frequency
    # Need at least HISTORY_LEN obs frames per call so RealEnv's deque
    # carries enough context for the proprio history we build.
    n_obs_steps = HISTORY_LEN
    print(f"n_obs_steps (matches HISTORY_LEN): {n_obs_steps}")
    print("Policy outputs single-step actions (n_action_steps=1)")

    side_depth_mean = 0.0
    side_depth_std = 0.0
    wrist_depth_mean = 0.0
    wrist_depth_std = 0.0
    timestep = 0

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shm_manager:
        with DA3DepthClient(device=0, model_id=da3_model_id,
                            process_res=da3_process_res if use_nested else None) as da3_client, \
             Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=(640, 480),
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                enable_depth=True,
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

            if use_da3_fusion:
                _ready_timeout = 180.0 if use_nested else 120.0
                print(f"Waiting for DA3 model to be ready (timeout={_ready_timeout:.0f}s)...")
                da3_client.wait_ready(timeout=_ready_timeout)
                print('DA3 model ready.')

            # Depth scales + intrinsics — read once after cameras are ready.
            side_depth_scale  = env.realsense.cameras[SIDE_SERIAL].get_depth_scale()
            wrist_depth_scale = env.realsense.cameras[WRIST_SERIAL].get_depth_scale()
            side_K  = env.realsense.cameras[SIDE_SERIAL].get_intrinsics()
            wrist_K = env.realsense.cameras[WRIST_SERIAL].get_intrinsics()
            SIDE_FOCAL_PX  = float(side_K[0, 0] + side_K[1, 1]) / 2.0
            WRIST_FOCAL_PX = float(wrist_K[0, 0] + wrist_K[1, 1]) / 2.0
            print(f'Depth scales — side: {side_depth_scale:.5f}, wrist: {wrist_depth_scale:.5f}')
            print(f'Focal lengths — side: {SIDE_FOCAL_PX:.1f} px, wrist: {WRIST_FOCAL_PX:.1f} px')

            # For NESTED, push intrinsics into shared memory once (used for pose conditioning)
            if use_nested and use_da3_fusion:
                da3_client.set_intrinsics(side_K, wrist_K)

            # ── DA3 background inference thread ───────────────────────────
            # Continuously processes queued RGB frames so the control loop
            # can read pre-computed depth without blocking on DA3 inference.
            da3_bg: DA3DepthBackground | None = None
            if use_da3_fusion:
                da3_bg = DA3DepthBackground(da3_client, SIDE_FOCAL_PX, WRIST_FOCAL_PX)
                da3_bg.start()
                print('DA3 background inference thread started.')

            # ── Depth video writer (mirrors demo_real_robot.py panels) ─────
            depth_video_writer = None
            depth_raw_video_writer = None
            rs_debug_video_writer = None
            da3_debug_video_writer = None
            if save_video:
                cv2.namedWindow('Depth (policy input)', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Depth (policy input)', DEPTH_IMG_W * 2, DEPTH_IMG_H)
                depth_video_path = pathlib.Path(output) / 'policy_depth_input.mp4'
                depth_video_writer = cv2.VideoWriter(
                    str(depth_video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    int(frequency),
                    (DEPTH_IMG_W * 2, DEPTH_IMG_H),
                )
                print(f'Policy-input depth video → {depth_video_path}')
                depth_raw_video_path = pathlib.Path(output) / 'policy_depth_raw.mp4'
                depth_raw_video_writer = cv2.VideoWriter(
                    str(depth_raw_video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    int(frequency),
                    (DEPTH_IMG_W * 2, DEPTH_IMG_H),
                )
                print(f'Policy-input raw depth video → {depth_raw_video_path}')
                rs_debug_path = pathlib.Path(output) / 'debug_realsense_raw.mp4'
                rs_debug_video_writer = cv2.VideoWriter(
                    str(rs_debug_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    int(frequency),
                    (DEPTH_IMG_W * 2, DEPTH_IMG_H),
                )
                print(f'RealSense raw depth video → {rs_debug_path}')
                if use_da3_fusion:
                    da3_debug_path = pathlib.Path(output) / 'debug_da3_metric.mp4'
                    da3_debug_video_writer = cv2.VideoWriter(
                        str(da3_debug_path),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        int(frequency),
                        (DEPTH_IMG_W * 2, DEPTH_IMG_H),
                    )
                    print(f'DA3 metric depth video → {da3_debug_path}')

            # ── Warm up policy with current obs (no inference latency in step 0) ──
            print("Warming up policy inference")
            obs = env.get_obs()
            obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])
            arm_jp_now = obs['arm_joint_pos'][-1]
            grip_now = float(obs['gripper_pos'][-1])
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
                    f"expects {expected_proprio}. Check joint_pos construction "
                    f"(NUM_JOINTS={NUM_JOINTS}) and the proprio cfg."
                )
            warmup_depth = torch.zeros(1, 1, DEPTH_IMG_H, DEPTH_IMG_W, device=device)
            with torch.no_grad():
                action_mean = policy(warmup_proprio, [warmup_depth, warmup_depth])
                print(f"Warmup OK; action shape={tuple(action_mean.shape)}")
                del action_mean

            print('Ready!')
            time.sleep(1.0)

            # Initialize concatenated camera video recording if enabled
            video_fps = int(frequency)
            episode_video_writer = None
            long_video_writer = None
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
            GRIPPER_OPEN_DURATION = 5
            STUCK_WINDOW_S = 2.0
            STUCK_JOINT_THRESHOLD_RAD = 0.002
            STUCK_GRIPPER_OPEN_STEPS = int(frequency)
            stuck_buffer = []

            # Per-episode state — reset on each new episode start.
            proprio_history: deque = deque(maxlen=HISTORY_LEN)
            last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

            while True:
                # ========== policy control loop ==============
                try:
                    proprio_history.clear()
                    last_raw_action = np.zeros(PREV_ACTION_DIM, dtype=np.float32)
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
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        # Override EE pose with calibrated FK (wrist_3_link in REP-103 frame)
                        obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])

                        # ── Queue latest frames for background DA3 inference ──
                        if use_da3_fusion:
                            _side_E = _wrist_E = None
                            if use_pose:
                                _arm_jp = obs['arm_joint_pos'][-1]
                                _pos_ee, _quat_ee = get_ee_pose(_arm_jp)
                                _wrist_E = _ee_to_w2c(_pos_ee, _quat_ee, wrist_cam_T)
                                _side_E  = side_cam_w2c
                            da3_bg.put_frame(obs['side_rgb'][-1], obs['wrist_rgb'][-1],
                                             side_extrinsic=_side_E,
                                             wrist_extrinsic=_wrist_E)

                        # Capture concatenated RGB frames if recording
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

                        # ── Build depth obs (mirrors sim's process_image) ──
                        if env.last_realsense_data is None:
                            raise RuntimeError("No realsense data yet — was RealEnv started with enable_depth=True?")
                        rs_side  = env.last_realsense_data[1].get('depth')
                        rs_wrist = env.last_realsense_data[2].get('depth')
                        if rs_side is None or rs_wrist is None:
                            raise RuntimeError("Depth frames missing from realsense buffer.")

                        da3_side_m = da3_wrist_m = None
                        if use_da3_fusion:
                            _da3_result = da3_bg.get_latest_da3()
                            if _da3_result is not None:
                                da3_side_m, da3_wrist_m, _ = _da3_result
                                if da3_only:
                                    side_norm  = _process_metric_depth(da3_side_m)
                                    wrist_norm = _process_metric_depth(da3_wrist_m)
                                else:
                                    fused_side  = DA3DepthClient.fuse_with_realsense(da3_side_m,  rs_side[-1],  side_depth_scale)
                                    fused_wrist = DA3DepthClient.fuse_with_realsense(da3_wrist_m, rs_wrist[-1], wrist_depth_scale)
                                    side_norm  = _process_metric_depth(fused_side)
                                    wrist_norm = _process_metric_depth(fused_wrist)
                            else:
                                # No DA3 result yet (first frame) — fall back to raw RealSense.
                                side_norm  = _process_realsense_depth(rs_side[-1],  side_depth_scale)
                                wrist_norm = _process_realsense_depth(rs_wrist[-1], wrist_depth_scale)
                        else:
                            side_norm  = _process_realsense_depth(rs_side[-1],  side_depth_scale)
                            wrist_norm = _process_realsense_depth(rs_wrist[-1], wrist_depth_scale)

                        # Visualise depth panels (mirrors demo_real_robot.py overlays)
                        side_vis  = _depth_to_bgr(side_norm)
                        wrist_vis = _depth_to_bgr(wrist_norm)
                        side_color  = cv2.resize(obs['side_rgb'][-1][:, :, ::-1],  (DEPTH_IMG_W, DEPTH_IMG_H))
                        wrist_color = cv2.resize(obs['wrist_rgb'][-1][:, :, ::-1], (DEPTH_IMG_W, DEPTH_IMG_H))
                        if side_color.dtype != np.uint8:
                            side_color = (np.clip(side_color, 0.0, 1.0) * 255).astype(np.uint8)
                        if wrist_color.dtype != np.uint8:
                            wrist_color = (np.clip(wrist_color, 0.0, 1.0) * 255).astype(np.uint8)
                        side_overlay  = cv2.addWeighted(side_vis,  0.5, side_color,  0.5, 0)
                        wrist_overlay = cv2.addWeighted(wrist_vis, 0.5, wrist_color, 0.5, 0)
                        cv2.putText(side_overlay,  'SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        cv2.putText(wrist_overlay, 'WRIST', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        depth_panel = np.concatenate([side_overlay, wrist_overlay], axis=1)
                        if depth_video_writer is not None:
                            depth_video_writer.write(depth_panel)
                        if depth_raw_video_writer is not None:
                            side_vis_raw  = _depth_to_bgr(side_norm).copy()
                            wrist_vis_raw = _depth_to_bgr(wrist_norm).copy()
                            cv2.putText(side_vis_raw,  'SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            cv2.putText(wrist_vis_raw, 'WRIST', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            depth_raw_panel = np.concatenate([side_vis_raw, wrist_vis_raw], axis=1)
                            depth_raw_video_writer.write(depth_raw_panel)
                        if rs_debug_video_writer is not None:
                            _wh = (DEPTH_IMG_W, DEPTH_IMG_H)
                            rs_s_m = rs_side[-1].astype(np.float32) * side_depth_scale
                            rs_w_m = rs_wrist[-1].astype(np.float32) * wrist_depth_scale
                            rs_s_m[rs_side[-1] == 0] = DEPTH_CLIP[1]
                            rs_w_m[rs_wrist[-1] == 0] = DEPTH_CLIP[1]
                            rs_s_vis = _metric_to_bgr(rs_s_m, _wh)
                            rs_w_vis = _metric_to_bgr(rs_w_m, _wh)
                            cv2.putText(rs_s_vis, 'RS SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            cv2.putText(rs_w_vis, 'RS WRIST', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            rs_debug_video_writer.write(np.concatenate([rs_s_vis, rs_w_vis], axis=1))
                        if da3_debug_video_writer is not None and use_da3_fusion and da3_side_m is not None:
                            _wh = (DEPTH_IMG_W, DEPTH_IMG_H)
                            da3_s_vis = _metric_to_bgr(da3_side_m,  _wh)
                            da3_w_vis = _metric_to_bgr(da3_wrist_m, _wh)
                            cv2.putText(da3_s_vis, 'DA3 SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            cv2.putText(da3_w_vis, 'DA3 WRIST', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            da3_debug_video_writer.write(np.concatenate([da3_s_vis, da3_w_vis], axis=1))

                        # ── Build proprio frame and append to history ──
                        arm_jp_now = obs['arm_joint_pos'][-1]
                        grip_pos_raw = float(obs['gripper_pos'][-1])
                        ee_pose_now  = obs['end_effector_pose'][-1].astype(np.float32)
                        proprio_history.append({
                            "prev_action": last_raw_action.copy(),
                            "joint_pos":   _build_joint_pos(arm_jp_now, grip_pos_raw),
                            "ee_pose":     ee_pose_now,
                        })
                        proprio_tensor = _build_proprio_tensor(proprio_history, device)
                        side_t  = torch.from_numpy(side_norm).to(device)[None, None]   # (1,1,H,W)
                        wrist_t = torch.from_numpy(wrist_norm).to(device)[None, None]

                        side_depth_mean += side_norm.mean()
                        side_depth_std += side_norm.std()
                        wrist_depth_mean += wrist_norm.mean()
                        wrist_depth_std += wrist_norm.std()
                        timestep += 1
                        # ── Run inference ──
                        with torch.no_grad():
                            action_mean = policy(proprio_tensor, [side_t, wrist_t]).cpu().numpy()
                        # action_mean: (1, num_actions=7) — [arm_delta(6), gripper(1)]
                        raw_action = action_mean[0]  # (7,)
                        if action_noise > 0:
                            raw_action[:6] = raw_action[:6] + np.random.randn(6) * action_noise

                        # Stuck detection (joint range over STUCK_WINDOW_S)
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
                                    print("[Stuck detection] No movement for 2s, opening gripper")

                        # Gripper open macro
                        gripper_action = raw_action[6:7].copy()
                        if gripper_open_steps_remaining > 0:
                            gripper_action = np.array([1.0], dtype=np.float32)
                            gripper_open_steps_remaining -= 1
                            if gripper_open_steps_remaining == 0:
                                print("[Gripper macro] done, returning to policy control")
                        # Persist (possibly-overridden) action so next step's prev_action
                        # mirrors what the action manager would store in sim.
                        last_raw_action = np.concatenate([raw_action[:6], gripper_action]).astype(np.float32)

                        # ── Cartesian OSC: scale delta, compute absolute target ──
                        scaled_delta = raw_action[:6] * CARTESIAN_SCALE
                        obs_pos, obs_quat = get_ee_pose(arm_jp_now)
                        tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled_delta)
                        tgt_aa = quat_to_axis_angle(tgt_quat)
                        abs_target = np.concatenate([tgt_pos, tgt_aa])[None]  # (1, 6)
                        target_actions = np.concatenate([abs_target, gripper_action[None]], axis=1)
                        raw_actions    = np.concatenate([raw_action[:6][None], gripper_action[None]], axis=1)

                        if collect_sysid:
                            sysid_records.append((arm_jp_now.copy(), tgt_pos.copy(), tgt_quat.copy()))

                        # ── Schedule action ──
                        action_timestamp = float(obs_timestamps[-1] + dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        if action_timestamp <= curr_time + action_exec_latency:
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + next_step_idx * dt
                        action_timestamps = np.array([action_timestamp])

                        actions.append(target_actions)
                        env.exec_actions(
                            actions=target_actions,
                            timestamps=action_timestamps,
                            obs_actions=raw_actions,
                        )

                        # Visualise camera feed for key detection
                        episode_id = env.replay_buffer.n_episodes
                        camera_key = 'side_rgb'
                        if camera_key in obs:
                            vis_img = obs[camera_key][-1]
                            text = f'Episode: {episode_id}, Time: {time.monotonic() - t_start:.1f}'
                            cv2.putText(
                                vis_img, text, (10, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, color=(255, 255, 255)
                            )
                            cv2.imshow('Policy Control', vis_img[..., ::-1])
                        if save_video:
                            cv2.imshow('Depth (policy input)', depth_panel)
                        cv2.waitKey(1)  # refresh display only — key detection is via key_reader

                        key_stroke = key_reader.get()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening gripper for {GRIPPER_OPEN_DURATION} steps")
                        elif key_stroke == ord('s'):
                            save_sysid_data()
                            env.end_episode()
                            print('Stopped.')
                            print(f"Side depth mean: {side_depth_mean / timestep}, std: {side_depth_std / timestep}")
                            print(f"Wrist depth mean: {wrist_depth_mean / timestep}, std: {wrist_depth_std / timestep}")
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
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            precise_wait(eval_t_start, time_func=time.time)
                            iter_idx = 0
                            term_area_start_timestamp = float('inf')
                            print('Robot reset complete! Starting new trajectory.')
                            timestep = 0
                            side_depth_mean = 0.0
                            side_depth_std = 0.0
                            wrist_depth_mean = 0.0
                            wrist_depth_std = 0.0
                            print(f"Side depth mean: {side_depth_mean / timestep}, std: {side_depth_std / timestep}")
                            print(f"Wrist depth mean: {wrist_depth_mean / timestep}, std: {wrist_depth_std / timestep}")
                            continue

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            save_sysid_data()
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    save_sysid_data()
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
                    if depth_video_writer is not None:
                        depth_video_writer.release()
                    if depth_raw_video_writer is not None:
                        depth_raw_video_writer.release()
                    if rs_debug_video_writer is not None:
                        rs_debug_video_writer.release()
                    if da3_debug_video_writer is not None:
                        da3_debug_video_writer.release()
                    if da3_bg is not None:
                        da3_bg.stop()
                    break

                print("Stopped.")
                key_reader.stop()
                if save_video and episode_video_writer is not None:
                    episode_video_writer.close()
                    episode_video_writer = None
                if save_video and long_video_writer is not None:
                    long_video_writer.close()
                    long_video_writer = None
                    print(f"  Continuous video saved.")
                if depth_video_writer is not None:
                    depth_video_writer.release()
                if depth_raw_video_writer is not None:
                    depth_raw_video_writer.release()
                if rs_debug_video_writer is not None:
                    rs_debug_video_writer.release()
                if da3_debug_video_writer is not None:
                    da3_debug_video_writer.release()


# %%
if __name__ == '__main__':
    main()
