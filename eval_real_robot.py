"""
Evaluate a low-dim STATE policy (RSL-RL, TorchScript-exported) on the real UR5e.

The policy is the "basic RL" state policy from UWLab ``rl_state_cfg.py`` (PolicyCfg):
5 stacked frames of the concatenated state vector, in this exact term order --

    insertive_in_receptive (6)  peg in peg-hole frame
    prev_actions (7)  [dx,dy,dz,drx,dry,drz, gripper]   raw policy output, pre-scale
    joint_pos   (12)  6 arm joints + 6 Robotiq mimic joints (absolute)
    end_effector_pose (6)  wrist_3 in base frame     [pos, axis-angle]
    insertive_asset_pose (6)  peg in wrist_3 frame   [pos, axis-angle]
    receptive_asset_pose (6)  peg-hole in wrist_3 frame

The annotated ``insertive_asset_in_receptive_asset_frame`` configclass field is collected
before the unannotated fields. Each term is flattened oldest->newest, then the term blocks
are concatenated -> (6+7+12+6+6+6)*5 = 215 dims.

The peg (insertive) pose is produced by the DECOUPLED, tried-and-tested fusion worker
``scripts/sim2real/perception/peg_fusion_viz.py --publish --headless``, which owns the cameras
at their calibrated resolutions (front 1280x720, side/wrist 1920x1080), uses the per-serial
intrinsics WITH distortion, and runs the full fusion (prior-hold, single-tag exclusion, robust
merge) at its OWN high rate. It atomically publishes the fused ``T_base_peg`` to a JSON state
file; this control loop only FETCHES the latest pose (non-blocking) -- pose estimation never
blocks control. This eval therefore runs ``RealEnv`` CAMERA-LESS (robot control only).

The peg-hole (receptive) pose is LATCHED ONCE PER EPISODE and then held fixed: the same worker
publishes ``T_base_hole``, accumulated from the hole's own AprilTags over many frames/cameras
(the hole is bolted down, so averaging is free accuracy and freezing it keeps tag jitter out of
the receptive obs). Calibrate the hole tags once with
``scripts/sim2real/perception/calibrate_hole_tags.py``. Without that, or with
``--no_hole_from_tags``, it falls back to a fixed pose (``--hole_pose`` / ``--hole_pose_file``,
default hardcoded). The arm uses the EVAL RelCartesianOSC scaling (0.01,0.01,0.002,0.02,0.02,0.2)
+ binary gripper, matching ``Ur5eRobotiq2f85RelativeOSCEvalAction``.

Usage (two processes):
  # 1) pose worker (owns cameras) -- or pass --launch_fusion to this script to auto-start it:
  $ python scripts/sim2real/perception/peg_fusion_viz.py --publish --headless --robot_ip 192.168.1.10
  # 2) the eval:
  $ python eval_real_robot.py -i state_policy.pt -o <save_dir> --robot_ip 192.168.1.10

Controls (click the OpenCV window first):
  'C' start policy control   'S' stop / hand back    'R' reset + relabel + restart
  'G' open gripper (macro)   'Q' quit
Episodes auto-terminate on EE z > --z_terminate or --max_duration; then label 's'/'f'.

The history-conditioned tactile/in-context eval now lives in eval_real_robot_tactile.py.
"""

# %%
import time
import json
import pathlib
from collections import deque
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.ur5e_kinematics import (
    get_ee_pose, quat_to_axis_angle, apply_delta_pose, forward_kinematics_calibrated)

import imageio

# ---- state-obs layout (must byte-match rl_state_cfg.py PolicyCfg) --------------------
HISTORY_LEN = 5
PREV_ACTION_DIM = 7          # 6 OSC delta + 1 gripper
NUM_ARM_JOINTS = 6
NUM_GRIPPER_JOINTS = 6
NUM_JOINTS = NUM_ARM_JOINTS + NUM_GRIPPER_JOINTS   # 12
POSE_DIM = 6                 # pos(3) + axis-angle(3)
# Configclass collects the sole annotated PolicyCfg field first, followed by the unannotated
# fields in source order. This is the actual order consumed by Isaac's ObservationManager.
STATE_TERM_ORDER = (
    'in_receptive',  # insertive_asset_in_receptive_asset_frame (6)
    'prev_action',   # prev_actions (7)
    'joint_pos',     # joint_pos (12)
    'ee_pose',       # end_effector_pose (6)
    'insertive',     # insertive_asset_pose (6)
    'receptive',     # receptive_asset_pose (6)
)
STATE_OBS_DIM = (PREV_ACTION_DIM + NUM_JOINTS + POSE_DIM * 4) * HISTORY_LEN   # 215

# Eval RelCartesianOSC action scaling (Ur5eRobotiq2f85RelativeOSCEvalAction): raw policy
# arm output (axis-angle delta) -> metres / radians. Matches sim eval action space exactly.
CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])

# Robotiq 2F85 mimic reconstruction (matches eval_real_robot_depth.py). The controller
# normalizes gripper_pos to [0,1] (0=open -> master 0 rad, 1=closed -> master pi/4 rad);
# the 6 gripper DOFs Isaac returns are +/- that master angle in articulation order.
GRIPPER_POS_OPEN = 0.0
GRIPPER_POS_CLOSE = 1.0
GRIPPER_POS_TO_RAD = np.pi / 4 / (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN)
GRIPPER_MIMIC_RATIOS = np.array([+1.0, +1.0, -1.0, +1.0, -1.0, -1.0], dtype=np.float32)

# Fixed peg-hole (receptive) pose in the REP-103 base frame, [x,y,z, qw,qx,qy,qz]. Used when
# neither --hole_pose nor --hole_pose_file is given. Identity quaternion = axis-aligned hole.
DEFAULT_HOLE_POSE = [0.5400, 0.225, -0.005, 1.0, 0.0, 0.0, 0.0]

# Default state file the peg_fusion_viz --publish worker writes the fused peg pose to
# (matches peg_fusion_viz.DEFAULT_STATE_FILE = <repo>/Log/peg_twin_state.json).
_REPO_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_PEG_STATE_FILE = str(_REPO_DIR / "Log" / "peg_twin_state.json")
PEG_FUSION_SCRIPT = str(_REPO_DIR / "scripts" / "sim2real" / "perception" / "peg_fusion_viz.py")

def compute_calibrated_ee_pose(joint_positions: np.ndarray) -> np.ndarray:
    """EE pose (wrist_3_link in REP-103 base) as (n,6) [x,y,z, rx,ry,rz] axis-angle.

    Isaac Lab's ``axis_angle_from_quat`` first maps q and -q to the representative with
    non-negative scalar component.  SciPy's ``as_rotvec`` uses that same canonical,
    shortest-angle representation.  The older real helper ``quat_to_axis_angle`` does not
    canonicalize quaternion sign and can therefore differ from Isaac by 2*pi when q.w < 0.
    """
    n_steps = joint_positions.shape[0]
    ee_poses = np.zeros((n_steps, POSE_DIM), dtype=np.float32)
    for t in range(n_steps):
        pos, quat = get_ee_pose(joint_positions[t])
        ee_poses[t, :3] = pos
        # get_ee_pose returns [w,x,y,z]; scipy expects [x,y,z,w].
        ee_poses[t, 3:] = R.from_quat(quat[[1, 2, 3, 0]]).as_rotvec()
    return ee_poses

def _load_jit_metadata(jit_path):
    """Read sidecar metadata (<stem>_meta.txt) written by the policy exporter, if present."""
    meta_path = pathlib.Path(jit_path).with_name(pathlib.Path(jit_path).stem + "_meta.txt")
    meta = {}
    if not meta_path.exists():
        print(f"[WARN] no sidecar metadata at {meta_path}; skipping shape validation.")
        return meta
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    print(f"[loaded JIT metadata] {meta}")
    return meta


def _last_scalar(x):
    """Most-recent scalar from a (T,1)/(T,)/scalar obs entry."""
    return float(np.asarray(x).reshape(-1)[-1] if np.asarray(x).ndim else np.asarray(x))


def _inv(T):
    """Fast rigid 4x4 inverse."""
    Ti = np.eye(4)
    Rt = T[:3, :3].T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ T[:3, 3]
    return Ti


def _mat_to_pos_aa(T):
    """4x4 -> [x,y,z, aa_x,aa_y,aa_z] (axis-angle == rotvec; matches sim axis_angle_from_quat)."""
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()]).astype(np.float32)


def _build_joint_pos(arm_joint_pos, gripper_pos_raw):
    """6 arm joints + 6 reconstructed Robotiq mimic joints -> (12,) float32 (absolute)."""
    arm = np.asarray(arm_joint_pos, dtype=np.float32).reshape(-1)[:NUM_ARM_JOINTS]
    master_angle = (float(gripper_pos_raw) - GRIPPER_POS_OPEN) * GRIPPER_POS_TO_RAD
    gripper = (GRIPPER_MIMIC_RATIOS * master_angle).astype(np.float32)
    return np.concatenate([arm, gripper], axis=0)


def _build_state_obs(history, device):
    """Stack the 5-frame history into the (1, 215) policy input.

    Layout matches Isaac Lab ObservationManager (concatenate_terms=True,
    flatten_history_dim=True, history_length=5): each term is flattened over its 5-frame
    history (oldest->newest) and the flattened blocks are concatenated in cfg term order.
    Short histories are back-filled by repeating the earliest frame (CircularBuffer behavior).
    """
    h = list(history)
    while len(h) < HISTORY_LEN:
        h.insert(0, h[0])
    terms = [np.concatenate([f[k] for f in h], axis=0) for k in STATE_TERM_ORDER]
    flat = np.concatenate(terms, axis=0).astype(np.float32)
    return torch.from_numpy(flat).unsqueeze(0).to(device)


def _fmt6(v):
    """Format a 6-vec [pos(3), axis-angle(3)] as pos=[..] aa=[..] |aa|=deg."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    aa_deg = np.degrees(np.linalg.norm(v[3:6]))
    return (f"pos[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}] "
            f"aa[{v[3]:+.3f} {v[4]:+.3f} {v[5]:+.3f}] (|aa|={aa_deg:5.1f}deg)")


def _debug_print_obs(frame, peg_seen):
    """One-line-per-block dump of the per-frame state obs, for OOD debugging."""
    tag = "" if peg_seen else "  <-- PEG UNSEEN (zeros)"
    print(f"  [obs] in_receptive  {_fmt6(frame['in_receptive'])}{tag}   (peg in hole)")
    print("  [obs] prev_action ", np.array2string(np.asarray(frame['prev_action']),
                                                   precision=3, suppress_small=True))
    print("  [obs] joint_pos    ", np.array2string(np.asarray(frame['joint_pos']),
                                                    precision=3, suppress_small=True))
    print(f"  [obs] ee_pose       {_fmt6(frame['ee_pose'])}")
    print(f"  [obs] insertive     {_fmt6(frame['insertive'])}{tag}   (peg in wrist; should be small+stable)")
    print(f"  [obs] receptive     {_fmt6(frame['receptive'])}   (hole in wrist)")


def _write_record_ctrl(path, recording, episode_id, video_dir):
    """Atomically write the record-control file the peg_fusion_viz worker polls to know when
    (recording) and where (video_dir/{episode_id}/{cam_idx}.mp4) to save the camera videos."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps({'stamp': time.time(), 'recording': bool(recording),
                               'episode_id': int(episode_id), 'video_dir': str(video_dir)}))
    tmp.replace(path)


def _load_hole_pose(hole_pose_file, hole_pose):
    """Fixed peg-hole (receptive) pose in the REP-103 base frame, as a 4x4.

    --hole_pose_file JSON: either {"T_base_hole": 4x4} or {"pos":[x,y,z], "quat_wxyz":[w,x,y,z]}.
    --hole_pose "x,y,z,rx,ry,rz": position + axis-angle.
    """
    if hole_pose_file:
        d = json.load(open(hole_pose_file))
        if "T_base_hole" in d:
            return np.array(d["T_base_hole"], dtype=np.float64)
        pos = np.asarray(d["pos"], dtype=np.float64)
        quat = np.asarray(d.get("quat_wxyz", d.get("quat")), dtype=np.float64)  # [w,x,y,z]
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()  # -> [x,y,z,w] for scipy
        T[:3, 3] = pos
        return T
    if hole_pose:
        v = [float(x) for x in hole_pose.replace(" ", "").split(",")]
        if len(v) != 6:
            raise SystemExit("--hole_pose must be 'x,y,z,rx,ry,rz' (6 numbers, axis-angle).")
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(v[3:6]).as_matrix()
        T[:3, 3] = v[:3]
        return T
    # default: hardcoded [x,y,z, qw,qx,qy,qz]
    p = DEFAULT_HOLE_POSE
    T = np.eye(4)
    T[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()  # [w,x,y,z] -> scipy [x,y,z,w]
    T[:3, 3] = p[:3]
    print(f"[hole] using hardcoded default peg-hole pose {p}")
    return T


@click.command()
@click.option('--input', '-i', required=True, help='Path to TorchScript state policy (.pt).')
@click.option('--output', '-o', required=True, help='Directory to save recording/results.')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP e.g. 192.168.1.10")
@click.option('--peg_state_file', default=DEFAULT_PEG_STATE_FILE, type=str,
              help="JSON state file the peg_fusion_viz --publish worker writes the fused peg "
                   f"pose to. Default {DEFAULT_PEG_STATE_FILE}.")
@click.option('--peg_stale_s', default=0.5, type=float,
              help="Treat the published peg pose as unseen if its stamp is older than this (s).")
@click.option('--launch_fusion', is_flag=True, default=False,
              help="Auto-launch peg_fusion_viz.py --publish --headless as a background worker "
                   "(owns the cameras) and terminate it on exit. Off by default -- normally you "
                   "run the worker yourself in a separate terminal.")
@click.option('--fusion_extra', default='', type=str,
              help="Extra args appended to the auto-launched worker (e.g. \"--fuse-exclude front\").")
@click.option('--hole_pose_file', default=None, type=str,
              help='JSON with the fixed peg-hole base pose ({"T_base_hole":4x4} or '
                   '{"pos":[..],"quat_wxyz":[..]}).')
@click.option('--hole_pose', default=None, type=str,
              help='Fixed peg-hole base pose as "x,y,z,rx,ry,rz" (axis-angle).')
@click.option('--hole_from_tags/--no_hole_from_tags', default=True,
              help='Latch the peg-hole pose from the fusion worker\'s AprilTag estimate at the '
                   'start of every episode, then hold it FIXED for the whole episode (the hole '
                   'does not move mid-episode, and a frozen pose cannot jitter the receptive '
                   'obs). Requires calibrate_hole_tags.py + a worker running with --hole-tags. '
                   'Falls back to --hole_pose_file / --hole_pose / the hardcoded default when no '
                   'tag-based pose is published. ON by default.')
@click.option('--hole_min_samples', default=8, type=int,
              help='Require at least this many accumulated hole-tag observations before trusting '
                   'the published pose over the configured fallback.')
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Initialize robot joint configuration at startup.")
@click.option('--max_duration', '-md', default=90, help='Max episode duration (s).')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency (Hz).")
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise added to raw arm actions (pre-scale).')
@click.option('--z_terminate', default=0.4, type=float,
              help='Auto-terminate when EE z (base frame, m) exceeds this.')
@click.option('--save_video/--no_save_video', default=True,
              help='Record per-episode video (ON by default). Writes the status canvas to '
                   'policy_status_ep_NNN.mp4 AND, via a record-control file, tells the running '
                   'peg_fusion_viz worker to save the CAMERA videos to output/videos/{ep}/{idx}.mp4 '
                   '(the same layout RealEnv-with-cameras would use). --no_save_video disables both.')
@click.option('--debug_obs', is_flag=True, default=False,
              help='Print the per-frame state obs split into named blocks (~1 Hz): prev_action, '
                   'joint_pos, ee_pose, insertive, receptive, in_receptive. Use to eyeball '
                   'grasp-invariance (insertive should be small + stable) and spot OOD blocks.')
@click.option('--torch_device', default='cuda', type=str, help='Torch device for inference.')
def main(input, output, robot_ip, peg_state_file, peg_stale_s, launch_fusion, fusion_extra,
         hole_pose_file, hole_pose, hole_from_tags, hole_min_samples, init_joints, max_duration,
         frequency, action_noise, z_terminate, save_video, debug_obs, torch_device):

    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # RealEnv would put camera videos under output/videos/{episode}/{cam_idx}.mp4; the fusion
    # worker records there via the record-control file (since it -- not this env -- owns cameras).
    video_dir = output / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    record_ctrl_path = video_dir / 'record_control.json'
    if save_video:
        _write_record_ctrl(record_ctrl_path, False, 0, video_dir)   # idle until an episode starts
    device = torch.device(torch_device if torch.cuda.is_available() else 'cpu')

    # ---- peg-hole (receptive) pose ----
    # Held in a mutable cell: with --hole_from_tags it is re-latched from the fusion worker's
    # AprilTag estimate at the start of each episode and then FROZEN for that episode's duration.
    hole_fallback = _load_hole_pose(hole_pose_file, hole_pose)
    hole = {'T': hole_fallback, 'T_inv': _inv(hole_fallback), 'src': 'configured fallback'}
    print(f"Peg-hole (receptive) base pose [{hole['src']}]:\n"
          f"{np.array2string(hole['T'], precision=4)}")

    # ---- load TorchScript state policy ----
    policy = torch.jit.load(input, map_location=device).eval()
    meta = _load_jit_metadata(input)
    expected = int(meta.get('num_proprio', 0))
    if expected and expected != STATE_OBS_DIM:
        raise SystemExit(
            f"proprio dim mismatch: policy metadata num_proprio={expected} but this eval "
            f"builds {STATE_OBS_DIM}. Check the obs layout / cfg before running.")
    print(f"State obs dim: {STATE_OBS_DIM}  |  term order: {STATE_TERM_ORDER}"
          f"  |  Eval OSC scale: {CARTESIAN_SCALE}")

    # ---- peg-pose worker: the cameras + tested fusion run in a SEPARATE process
    # (peg_fusion_viz.py --publish --headless); this eval only reads the latest pose. ----
    import subprocess
    fusion_proc = None
    if launch_fusion:
        cmd = ['python', PEG_FUSION_SCRIPT, '--publish', peg_state_file, '--headless',
               '--robot_ip', robot_ip]
        if save_video:
            cmd += ['--record-control', str(record_ctrl_path)]
        if fusion_extra:
            cmd += fusion_extra.split()
        print(f"[fusion] launching worker: {' '.join(cmd)}")
        fusion_proc = subprocess.Popen(cmd)
        import atexit
        atexit.register(lambda: fusion_proc and fusion_proc.poll() is None and fusion_proc.terminate())
        time.sleep(6.0)  # give it time to open cameras + publish the first pose
    else:
        rc = f" --record-control {record_ctrl_path}" if save_video else ""
        print("[fusion] expecting an already-running worker. If not, run in another terminal:\n"
              f"    python {PEG_FUSION_SCRIPT} --publish {peg_state_file} --headless "
              f"--robot_ip {robot_ip}{rc}")

    # ---- episode success/fail labeling (writes output/eval_results.json) ----
    episode_results = []

    def prompt_success_fail(episode_id, ee_z):
        print(f"Episode {episode_id} terminated (EE z={ee_z:.3f}). 's'=SUCCESS, 'f'=FAIL ...")
        result = None
        while result is None:
            canvas = np.zeros((200, 700, 3), dtype=np.uint8)
            cv2.putText(canvas, f"EPISODE {episode_id} TERMINATED (z={ee_z:.3f})", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
        episode_results.append({'episode': int(episode_id),
                                'result': 'success' if result else 'fail',
                                'z': float(ee_z), 't': time.time()})
        n_succ = sum(r['result'] == 'success' for r in episode_results)
        n_tot = len(episode_results)
        with open(output / 'eval_results.json', 'w') as f:
            json.dump({'n_episodes': n_tot, 'n_success': n_succ,
                       'success_rate': n_succ / n_tot, 'episodes': episode_results}, f, indent=2)
        print(f"  [{n_succ}/{n_tot} success] saved to {output / 'eval_results.json'}")
        return result

    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
                output_dir=str(output),
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=2,
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                record_raw_video=True,
                rolling_action_buffer=True,
                action_mode='cartesian',
                arm_action_dim=6,               # 6-DoF Cartesian delta + binary gripper
                # Camera-less: the peg cameras + tested fusion run in the separate
                # peg_fusion_viz --publish worker; this env only controls the robot.
                camera_serial_numbers=[],
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)

            # Wait for the robot (RTDE) to finish connecting. With cameras this was masked by
            # the camera-init delay; camera-less we must wait explicitly or is_ready races False.
            print("Waiting for robot (RTDE) to be ready...")
            _t0 = time.time()
            while not env.is_ready:
                if time.time() - _t0 > 20.0:
                    raise SystemExit(
                        "robot not ready after 20s. Check the UR is powered, in REMOTE control, "
                        f"reachable at {robot_ip}, and not held by another RTDE control process.")
                time.sleep(0.2)
            print("Robot ready.")

            # Point get_obs_state at the worker's published peg pose + check it's live.
            env.setup_state_pose_reader(peg_state_file, stale_after_s=peg_stale_s)
            peg0 = env.state_pose_reader.read()
            if not peg0['seen']:
                print(f"[fusion][WARN] no fresh peg pose at {peg_state_file} "
                      f"(age={peg0['age_s']:.1f}s). Is the peg_fusion_viz --publish worker running "
                      "and seeing the peg? insertive/in_receptive obs will be zero until it is.")
            else:
                print(f"[fusion] peg pose live (age={peg0['age_s']:.2f}s, ncams={peg0['ncams']}, "
                      f"tag detections={peg0['ntag_detections']}, ids={peg0['tag_ids']}).")

            def latch_hole_pose(context=''):
                """Freeze the peg-hole pose for the episode that is about to start.

                The hole is bolted down: it cannot move mid-episode, so re-reading it every
                control step would only inject tag jitter into `receptive`/`in_receptive`.
                We take the worker's ACCUMULATED estimate (a robust average over many frames
                and cameras) once, print it, and hold it. If the fixture gets bumped between
                episodes, the next latch picks that up automatically.
                """
                if not hole_from_tags:
                    return
                st = env.state_pose_reader.read()
                T_new = st.get('T_base_hole')
                info = st.get('hole') or {}
                n = int(info.get('n', 0) or 0)
                if T_new is None or n < hole_min_samples:
                    why = 'nothing published' if T_new is None else f'only {n} observations'
                    if hole['src'] == 'configured fallback':
                        print(f"[hole]{context} no tag-based pose ({why}); using the "
                              f"{hole['src']}. Run calibrate_hole_tags.py and start the worker "
                              f"with --hole-tags to enable it.")
                    else:
                        print(f"[hole]{context} tag-based pose unavailable ({why}); "
                              f"HOLDING the previously latched pose.")
                    return
                prev = hole['T']
                hole['T'] = np.asarray(T_new, dtype=np.float64)
                hole['T_inv'] = _inv(hole['T'])
                hole['src'] = f"apriltags (n={n}, cams={info.get('cams')}, ids={info.get('tags')})"
                d_mm = np.linalg.norm(hole['T'][:3, 3] - prev[:3, 3]) * 1000.0
                d_deg = np.degrees(np.linalg.norm(R.from_matrix(
                    hole['T'][:3, :3] @ prev[:3, :3].T).as_rotvec()))
                print(f"[hole]{context} latched from {hole['src']}  "
                      f"pos={np.round(hole['T'][:3, 3], 4)}  "
                      f"spread={info.get('spread_mm', 0.0):.2f}mm/{info.get('spread_deg', 0.0):.2f}deg"
                      f"  (moved {d_mm:.1f}mm / {d_deg:.1f}deg since last latch)")

            latch_hole_pose(' startup:')

            # ---- warmup ----
            print("Warming up policy inference...")
            history = deque(maxlen=HISTORY_LEN)
            last_peg = [None]        # last valid T_base_peg (held through dropouts)

            def build_frame(obs):
                """Assemble one per-frame state dict from a get_obs_state() observation."""
                arm_jp = np.asarray(obs['arm_joint_pos'][-1], dtype=np.float64)[:NUM_ARM_JOINTS]
                grip_raw = _last_scalar(obs['gripper_pos'])
                T_base_ee = forward_kinematics_calibrated(arm_jp)[0]   # for the object transforms
                ee_inv = _inv(T_base_ee)
                # Use Isaac-compatible canonical axis-angle for the EE, just as the three
                # relative object-pose terms below do.
                ee_pose = compute_calibrated_ee_pose(arm_jp[None])[0]
                if obs.get('peg_seen') and obs.get('peg_pose_base') is not None:
                    last_peg[0] = np.asarray(obs['peg_pose_base'], dtype=np.float64)
                T_base_peg = last_peg[0]
                # prev_action = the raw action the env last executed (single source of truth,
                # tracked by RealEnv; never stalls). Zeros before the first action / at reset.
                prev_raw = env.get_last_action().astype(np.float32)
                receptive = _mat_to_pos_aa(ee_inv @ hole['T'])
                if T_base_peg is not None:
                    insertive = _mat_to_pos_aa(ee_inv @ T_base_peg)
                    in_receptive = _mat_to_pos_aa(hole['T_inv'] @ T_base_peg)
                else:
                    insertive = np.zeros(POSE_DIM, dtype=np.float32)
                    in_receptive = np.zeros(POSE_DIM, dtype=np.float32)
                return {
                    'in_receptive': in_receptive,
                    'prev_action': prev_raw,
                    'joint_pos': _build_joint_pos(arm_jp, grip_raw),
                    'ee_pose': ee_pose,
                    'insertive': insertive,
                    'receptive': receptive,
                }, T_base_ee, T_base_peg

            obs = env.get_obs(modality='state')
            frame, _, _ = build_frame(obs)
            history.append(frame)
            with torch.no_grad():
                out = policy(_build_state_obs(history, device))
                _ = (out[0] if isinstance(out, (tuple, list)) else out)
            print("Ready!")
            time.sleep(1.0)

            video_fps = int(frequency)
            GRIPPER_OPEN_DURATION = 5
            STUCK_WINDOW_S = 2.0
            STUCK_JOINT_THRESHOLD_RAD = 0.002
            STUCK_GRIPPER_OPEN_STEPS = int(frequency)
            peg_unseen_warned = [False]
            had_apriltags = [peg0['ntag_detections'] > 0]

            def set_recording(on):
                """Tell the peg_fusion_viz worker to start/stop saving camera videos for the
                current episode (output/videos/{episode}/{cam_idx}.mp4)."""
                if save_video:
                    ep = getattr(env.replay_buffer, 'n_episodes', 0)
                    _write_record_ctrl(record_ctrl_path, on, ep, video_dir)

            def open_status_video():
                if not save_video:
                    return None
                ep = getattr(env.replay_buffer, 'n_episodes', 0)
                return imageio.get_writer(
                    str(output / f'policy_status_ep_{ep:03d}.mp4'), fps=video_fps,
                    codec='libx264', output_params=['-crf', '21', '-preset', 'fast'])

            while True:
                # ================= policy control loop =================
                try:
                    print('Resetting robot to initial position...')
                    env.robot.reset_to_initial_position()
                    time.sleep(5.0)
                    print('Reset complete.')

                    latch_hole_pose(f" ep{env.replay_buffer.n_episodes}:")
                    history.clear()
                    last_peg[0] = None
                    gripper_open_steps_remaining = 0
                    stuck_buffer = []
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start, time_func=time.time)
                    print("Started!")

                    set_recording(True)                       # fusion worker records this episode
                    episode_video_writer = open_status_video()

                    iter_idx = 0
                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        obs = env.get_obs(modality='state')
                        obs_timestamps = obs['timestamp']
                        has_apriltags = obs.get('peg_ntag_detections', 0) > 0
                        if had_apriltags[0] and not has_apriltags:
                            print("[fusion][APRILTAG LOST] all cameras lost all AprilTags; "
                                  "holding the last valid peg pose.")
                        elif not had_apriltags[0] and has_apriltags:
                            print(f"[fusion][APRILTAG REACQUIRED] detections="
                                  f"{obs['peg_ntag_detections']} ids={obs.get('peg_tag_ids', [])}")
                        had_apriltags[0] = has_apriltags
                        frame, T_base_ee, T_base_peg = build_frame(obs)
                        history.append(frame)
                        if T_base_peg is None and not peg_unseen_warned[0]:
                            print("[WARN] peg not seen yet -- insertive/in_receptive terms are zero.")
                            peg_unseen_warned[0] = True
                        if debug_obs and iter_idx % max(1, int(frequency)) == 0:
                            print(f"[debug_obs] iter={iter_idx}  peg_seen={obs.get('peg_seen')}"
                                  f"  ncams={obs.get('peg_ncams')}  age={obs.get('peg_stamp_age_s', float('inf')):.2f}s")
                            _debug_print_obs(frame, bool(obs.get('peg_seen')))

                        # capture -> policy-input latency: from the oldest camera frame that
                        # produced this peg pose (RealSense read_time) to right now, as the policy
                        # is fed. inf until the first fresh publish. Throttled to ~1 Hz.
                        cap_stamp = obs.get('peg_capture_stamp')
                        cap_to_policy = (time.time() - cap_stamp) if cap_stamp is not None else float('inf')
                        if iter_idx % max(1, int(frequency)) == 0:
                            print(f"[latency] peg capture->policy = {cap_to_policy * 1e3:6.1f} ms"
                                  f"  (publish->read age={obs.get('peg_stamp_age_s', float('inf')) * 1e3:.1f} ms,"
                                  f" seen={obs.get('peg_seen')})")

                        with torch.no_grad():
                            out = policy(_build_state_obs(history, device))
                            action_mean = (out[0] if isinstance(out, (tuple, list)) else out)
                            action_mean = action_mean.detach().to('cpu').numpy().reshape(-1)

                        raw_arm = action_mean[:6].astype(np.float64)
                        raw_gripper = float(action_mean[6])   # raw policy output (>0 open, <0 close)
                        if action_noise > 0:
                            raw_arm = raw_arm + np.random.randn(6) * action_noise

                        # Stuck detection: if the arm barely moved for STUCK_WINDOW_S, open gripper.
                        arm_jp_now = np.asarray(obs['arm_joint_pos'][-1], dtype=np.float64)[:6]
                        if gripper_open_steps_remaining == 0:
                            t_now = time.monotonic()
                            stuck_buffer.append((t_now, arm_jp_now.copy()))
                            while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                stuck_buffer.pop(0)
                            if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                jps = np.array([b[1] for b in stuck_buffer])
                                if np.max(jps.max(0) - jps.min(0)) < STUCK_JOINT_THRESHOLD_RAD:
                                    gripper_open_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                    stuck_buffer.clear()
                                    print("[Stuck detection] no movement for 2s, opening gripper")

                        # Executed gripper (macro can force open); the SAVED raw action keeps the
                        # policy's own gripper output (user: "save the raw past action output by policy").
                        exec_gripper = raw_gripper
                        if gripper_open_steps_remaining > 0:
                            exec_gripper = 1.0
                            gripper_open_steps_remaining -= 1
                            if gripper_open_steps_remaining == 0:
                                print("[Gripper macro] done, returning to policy control")

                        # RelCartesian OSC: scale delta -> absolute EE target from observed pose.
                        scaled_delta = raw_arm * CARTESIAN_SCALE
                        obs_pos, obs_quat = get_ee_pose(arm_jp_now)
                        tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled_delta)
                        tgt_aa = quat_to_axis_angle(tgt_quat)
                        abs_target = np.concatenate([tgt_pos, tgt_aa])[None]          # (1,6)
                        target_actions = np.concatenate(
                            [abs_target, np.array([[exec_gripper]])], axis=1)         # (1,7)
                        raw_actions = np.concatenate(
                            [raw_arm[None], np.array([[raw_gripper]])], axis=1).astype(np.float32)

                        # timing -- schedule strictly in the future so exec_actions keeps the
                        # raw-action buffer flowing (it drops actions with past timestamps).
                        action_timestamps = np.array([obs_timestamps[-1] + dt], dtype=np.float64)
                        if action_timestamps[0] <= time.time() + 0.01:
                            next_step_idx = int(np.ceil((time.time() - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            while action_timestamps[0] <= time.time() + 0.01:
                                next_step_idx += 1
                                action_timestamps = np.array([eval_t_start + next_step_idx * dt])

                        # Execute; obs_actions = RAW policy output (pre-scale) -> prev_actions obs.
                        env.exec_actions(actions=target_actions, timestamps=action_timestamps,
                                         obs_actions=raw_actions)

                        # status canvas + key handling
                        ee_z = float(obs_pos[2])
                        episode_id = env.replay_buffer.n_episodes
                        canvas = np.zeros((240, 640, 3), dtype=np.uint8)
                        lines = [
                            f"Episode {episode_id}  t={time.monotonic() - t_start:5.1f}s",
                            f"peg seen={obs.get('peg_seen')}  ncams={obs.get('peg_ncams')}"
                            f"  tags={obs.get('peg_ntag_detections', 0)}"
                            f"  age={obs.get('peg_stamp_age_s', float('inf')):.2f}s",
                            f"EE z={ee_z:.3f}  gripper cmd={exec_gripper:+.2f}",
                            "S=stop  R=reset  G=open  Q=quit",
                        ]
                        for i, ln in enumerate(lines):
                            cv2.putText(canvas, ln, (12, 40 + 40 * i),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.imshow('Policy Control', canvas)
                        if save_video and episode_video_writer is not None:
                            episode_video_writer.append_data(canvas[..., ::-1])

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                        elif key_stroke == ord('q'):
                            raise KeyboardInterrupt
                        elif key_stroke == ord('s'):
                            env.end_episode()
                            set_recording(False)
                            print('Stopped.')
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                            break
                        elif key_stroke == ord('r'):
                            env.end_episode()
                            set_recording(False)
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                            env.robot.reset_to_initial_position()
                            prompt_success_fail(episode_id, ee_z)
                            time.sleep(5.0)
                            latch_hole_pose(f" ep{env.replay_buffer.n_episodes}:")
                            history.clear()
                            last_peg[0] = None
                            stuck_buffer.clear()
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            precise_wait(eval_t_start, time_func=time.time)
                            set_recording(True)
                            episode_video_writer = open_status_video()
                            iter_idx = 0
                            print('Robot reset complete! Starting new trajectory.')
                            continue

                        # auto termination
                        terminate = False
                        if ee_z > z_terminate:
                            terminate = True
                            print(f'Terminated: EE z={ee_z:.3f} > {z_terminate:.3f}')
                        elif time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by timeout!')
                        if terminate:
                            env.end_episode()
                            set_recording(False)
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                            prompt_success_fail(episode_id, ee_z)
                            print('Episode terminated; restarting.')
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except KeyboardInterrupt:
                    print("\nInterrupted!")
                    env.end_episode()
                    set_recording(False)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    env.end_episode()
                    set_recording(False)
                    break

                print("Stopped.")


# %%
if __name__ == '__main__':
    main()
