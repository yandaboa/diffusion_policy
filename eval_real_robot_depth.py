"""
Eval a depth-DAgger student (rsl_rl ``StudentTeacherVision``) on the real UR5e.

Loads a JIT-exported depth student (built by ``play.py`` via
``export_vision_student_as_jit``) and feeds it the same proprio + 2-camera depth
obs the sim wrapper produced at training time:

  proprio:   (1, num_proprio) float32  — concat of history-flattened
             [prev_actions, joint_pos, end_effector_pose] (history_length=5)
  side_depth, front_depth: (1, 1, 224, 224) float32 in [0,1], clipped at
             ``DEPTH_CLIP = (0.01, 1.0)`` m (sim2real_depth_cfg.py:83), with
             no-return pixels mapped to d_max. Both cameras are natively 4:3 and
             are squashed to 224x224, exactly as sim squashes its 320x240 render.

Cameras (raw sensor depth — no DA3 fusion):
  side  — RealSense D435, enumerated by RealEnv/MultiRealsense (camera_idx 0).
  front — Orbbec Femto Bolt, read on a background thread (RealEnv's camera
          plumbing is RealSense-only, so the Orbbec is opened separately here).

Runs in the ``foundstereo`` env, not ``robodiff_real``: the Orbbec needs
``pyorbbecsdk``, which only foundstereo has (same env as eval_real_robot_pc.py).

Usage:
(foundstereo)$ python eval_real_robot_depth.py -i <depth_policy_jit> -o <save_dir> --robot_ip <ip>

================ Policy in control ==============
The policy drives from the moment an episode starts; there is no SpaceMouse
teleop stage in this script. Press "S" to stop, "R" to reset the robot to its
initial joints, "G" to force the gripper open for a few steps (also
auto-triggered after 2 s of no motion).
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
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait

# The Orbbec backend we need lives at scripts/sim2real/perception/orbbec.py, but a
# plain ``import orbbec`` would resolve to the top-level ``orbbec/`` package instead
# (different API — that one yields point clouds, not depth images). Load it by path
# under a distinct module name to sidestep the collision.
import importlib.util as _ilu
_ORBBEC_PATH = pathlib.Path(__file__).parent / 'scripts' / 'sim2real' / 'perception' / 'orbbec.py'
_spec = _ilu.spec_from_file_location('_perception_orbbec', _ORBBEC_PATH)
_perception_orbbec = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_perception_orbbec)
gather_orbbec_cameras = _perception_orbbec.gather_orbbec_cameras

# Add imageio import for video saving
import imageio

# Calibrated FK matching simulation (wrist_3_link in REP-103 base_link frame)
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# ── Depth constants — must match sim2real_depth_cfg.py (the SideFront task) ──
# DEPTH_CLIP is verbatim from sim2real_depth_cfg.py:83. Both the clip bounds and
# the normalisation scale matter: sim maps depth as (d - d_lo) / (d_hi - d_lo),
# so a different d_lo shifts every pixel the policy sees.
DEPTH_CLIP = (0.01, 1.0)            # metres — sim2real_depth_cfg.py:83
DEPTH_IMG_H, DEPTH_IMG_W = 224, 224 # sim's IMG_H, IMG_W (sim2real_depth_cfg.py:79)

# Sim renders depth at 4:3 (RENDER_H, RENDER_W = 240, 320) and resizes to 224x224,
# i.e. it squashes 4:3 → 1:1 rather than cropping. Both real cameras are natively
# 4:3 (D435 640x480, Orbbec 1280x960), so resizing them straight to 224x224
# reproduces the same anisotropic squash. Do NOT centre-crop to square here.

# ── Cameras ────────────────────────────────────────────────────────────────
# The D435 side camera is the only RealSense enumerated, so MultiRealsense
# assigns it camera_idx 0. RealEnv's positional name table (real_env.get_obs)
# maps idx 0 → 'side_rgb' for a <3-camera setup, which is what we want.
# The front view comes from the Orbbec, which RealEnv cannot enumerate.
SIDE_SERIAL = '832112070487'        # RealSense D435 (side)
SIDE_CAM_IDX = 0

# Orbbec depth is uint16 millimetres (see scripts/sim2real/perception/orbbec.py),
# so the same u16→metres path as RealSense applies with a 1e-3 scale.
ORBBEC_DEPTH_SCALE = 0.001

# ── Proprio layout (must match DepthDAggerObservationsCfg.ProprioCfg) ───────
# Per-frame: prev_actions (7) + joint_pos (12) + end_effector_pose (6) = 25
# History length 5, terms concatenated: 5*7 + 5*12 + 5*6 = 125 dims total.
# --proprio_mode slim drops prev_actions and the 6 synthetic mimic gripper
# joints (which are reconstructed, not measured): 5*6 + 5*6 = 60 dims. Term
# ordering is otherwise unchanged (joint_pos frames, then ee_pose frames).
# --proprio_mode slim_gripper is `slim` plus a 1-d measured gripper close
# fraction (0=open, 1=closed) appended after ee_pose — the real counterpart of
# sim's task_mdp.gripper_close_progress term in the ...-ArmProprio-GripperAux-v0
# task (ProprioArmGripperProgressCfg). Layout: 5*6 (arm) + 5*6 (ee) + 5*1
# (gripper) = 65 dims, matching that cfg's per-term history-flattened order.
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
        # cbreak disables ECHO. If the process dies before stop() runs — e.g. a
        # Ctrl+C KeyboardInterrupt, which is a BaseException and so slips past the
        # loop's `except Exception` — the terminal is left with echo off and typed
        # input stops showing. Register the restore with atexit so it fires on any
        # exit path (normal, exception, or interrupt); stop() is idempotent.
        import atexit
        atexit.register(self.stop)
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


class OrbbecDepthBackground:
    """Continuously grab RGB+depth from the Orbbec on a background thread.

    ``OrbbecCamera.read_camera()`` blocks on ``wait_for_frames(500)`` and raises
    after a retry budget, so calling it inline would stall (or kill) the 10 Hz
    control loop on a dropped frame. This thread keeps only the newest frame;
    the loop reads it without blocking, tolerating a slightly stale depth map
    the same way the DA3 background thread did.

    Use as a context manager. ``__exit__`` must run: destroying a still-streaming
    pyorbbecsdk pipeline aborts the process in C++ ("terminate called without an
    active exception"), which would otherwise mask any real Python traceback.
    """

    def __init__(self, camera=None) -> None:
        self._cam = camera
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray, float] | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._n_errors = 0

    def __enter__(self) -> 'OrbbecDepthBackground':
        if self._cam is None:
            print('Opening Orbbec front camera...')
            cams = gather_orbbec_cameras(rgb=True, depth=True, align='rgb')
            if len(cams) == 0:
                raise RuntimeError('No Orbbec camera found — front_depth is unavailable.')
            if len(cams) > 1:
                print(f'[WARN] {len(cams)} Orbbec devices found; using serial '
                      f'{cams[0]._serial_number}.')
            self._cam = cams[0]
        self.start()
        try:
            self.wait_first_frame(timeout=20.0)
        except Exception:
            self.stop()
            raise
        rgb, depth, _ = self.get_latest()
        print(f'Orbbec ready — rgb {rgb.shape}, depth {depth.shape} {depth.dtype} (mm)')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> 'OrbbecDepthBackground':
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='OrbbecDepthBackground')
        self._thread.start()
        return self

    def stop(self) -> None:
        """Idempotent — may be reached from both the error path and __exit__."""
        if self._cam is None:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._cam.disable_camera()
        except Exception as e:
            print(f'[Orbbec] disable_camera failed: {e}')
        self._cam = None

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frame = self._cam.read_camera()
            except Exception as e:
                # Transient dropout: keep serving the last good frame rather than
                # taking down the control loop.
                self._n_errors += 1
                if self._n_errors % 10 == 1:
                    print(f'[Orbbec] read_camera failed ({self._n_errors}x): {e}')
                continue
            with self._lock:
                # capture_time = SDK's true capture stamp on the host clock
                # (global-timestamp fitter, else SDK receive time); read_time —
                # stamped after align+copy — only remains as a legacy fallback.
                self._latest = (frame['rgb'], frame['depth'],
                                frame.get('capture_time', frame['read_time']))

    def get_latest(self) -> 'tuple[np.ndarray, np.ndarray, float] | None':
        """Return (rgb HxWx3 uint8, depth HxW uint16 mm, capture_time) or None."""
        with self._lock:
            return self._latest

    def wait_first_frame(self, timeout: float = 20.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.get_latest() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(f'Orbbec produced no frame within {timeout:.0f}s')


def _last_scalar(arr) -> float:
    """Most recent value of a per-timestep obs field, as a Python float.

    ``get_obs()`` returns ``gripper_pos`` shaped (T, 1) (real_env builds it as
    ``np.asarray([gripper_pos_raw])`` per step), so ``arr[-1]`` is a shape-(1,)
    array, not a scalar. NumPy 2 refuses ``float()`` on that ("only 0-dimensional
    arrays can be converted to Python scalars") where NumPy 1.x allowed it — and
    this script runs on NumPy 2 in foundstereo. Flatten first, matching the idiom
    real_env itself uses. Handles (T,) and (T, 1) alike.
    """
    return float(np.asarray(arr).reshape(-1)[-1])


def _preflight_realsense(env, n_obs_steps: int, timeout: float = 15.0) -> None:
    """Fail fast, and legibly, if a RealSense isn't delivering frames.

    Without this, a camera that streams no frames surfaces much later as a bare
    ``AssertionError`` from ``shared_memory_ring_buffer.get_last_k`` (asserting
    ``k <= count`` against an empty buffer), which says nothing about the cause.

    The usual cause is a wedged D435: an unclean exit (e.g. a core dump) leaves
    the device unable to stream, and its capture process dies with "Frame didn't
    arrive within 5000". A hardware reset clears it — see the message below.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        dead = [s for s, c in env.realsense.cameras.items() if not c.is_alive()]
        if dead:
            raise RuntimeError(
                f"RealSense capture process for {dead} has died — it is not "
                f"streaming (look for \"Frame didn't arrive within 5000\" above).\n"
                f"    The camera is usually wedged after an unclean exit. Reset it:\n"
                f"      python -c \"import pyrealsense2 as rs; [d.hardware_reset() "
                f"for d in rs.context().query_devices()]\"; sleep 8\n"
                f"    Then re-run. Replugging the USB cable also works."
            )
        try:
            env.realsense.get(k=n_obs_steps)
            print(f'RealSense preflight OK — {n_obs_steps} frames buffered.')
            return
        except AssertionError:
            time.sleep(0.5)   # buffer still filling
    raise RuntimeError(
        f"RealSense buffered fewer than {n_obs_steps} frames in {timeout:.0f}s. "
        f"The camera is alive but slow or stalling; try the hardware reset above."
    )


def _resize_like_sim(depth_norm: np.ndarray) -> np.ndarray:
    """Resize normalised depth to 224×224 using sim's exact resampling op.

    Sim (``depth_augs.py`` line ~112) ends with::

        img = F.interpolate(img, size=output_size, mode="bilinear", antialias=True)

    so we call the identical torch op rather than approximate it with cv2.
    ``antialias=True`` is the load-bearing part — it low-pass filters before
    decimating. cv2.INTER_LINEAR does not, and we downscale far harder than sim
    (Orbbec 1280×960→224² is 5.7×, vs sim's 320×240→224² at 1.4×), so plain
    bilinear would hand the policy aliased high-frequency structure that never
    existed in training (measured: MAE 0.10 and std 0.21 vs sim's 0.14).

    Runs on CPU: ~0.2 ms at these sizes, and it keeps this helper usable from the
    visualisation path without touching the GPU.
    """
    t = torch.from_numpy(np.ascontiguousarray(depth_norm, dtype=np.float32))[None, None]
    t = torch.nn.functional.interpolate(
        t, size=(DEPTH_IMG_H, DEPTH_IMG_W), mode='bilinear', antialias=True)
    return t[0, 0].numpy()


def _process_u16_depth(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    """Raw u16 sensor depth → policy-input float32 [0,1] @ 224×224.

    Serves both cameras: RealSense z16 (``depth_scale`` from the sensor) and the
    Orbbec, whose depth is u16 millimetres (``ORBBEC_DEPTH_SCALE = 1e-3``). Both
    use 0 to mean "no return".

    Mirrors sim's ``dextrah_depth_image`` (mdp/depth_augs.py:360), which reads
    ``data_type='distance_to_image_plane'`` — z-depth, i.e. what a real RealSense
    or Orbbec reports — and then:
      1. Convert u16 → metres via ``depth_scale`` (m/unit).
      2. Map zero-pixels (no return) → ``d_max`` so they read as far range. Sim
         does the same: ``no_return = ~isfinite | (depth <= 0)`` is pinned to
         ``d_hi`` so missing pixels normalise to 1.0 (flat far).
      3. Clip to ``DEPTH_CLIP`` and normalise: ``(d - d_lo) / (d_hi - d_lo)``.
      4. Resize to 224×224 with ``_resize_like_sim`` — bilinear **with antialias**,
         the same op sim applies. Not cv2.INTER_LINEAR: that does no antialiasing
         when downscaling, and we shrink much harder than sim does (Orbbec 5.7×
         vs sim's 1.4×), which leaves aliasing sim never produced.

    Sim's augmentations (noise, dropout, sticks, corner crops, stereo shadow) are
    training-time only and deliberately not reproduced — they model the real
    sensor artefacts we already get for free.

    ``float(depth_scale)`` is load-bearing, not cosmetic. ``get_depth_scale()``
    returns a **np.float64** (its shm intrinsics_array is float64), and under
    NumPy 2's NEP 50 a NumPy scalar takes part in promotion: float32 * np.float64
    → float64, which reaches the policy as a DoubleTensor and raises "Input type
    (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be
    the same". NumPy 1.x kept it float32 via value-based casting, so this only
    bites on the NumPy 2 in foundstereo. Demoting to a Python float keeps the
    scalar "weak" and the result float32.
    """
    d_min, d_max = DEPTH_CLIP
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    depth_m[depth_m == 0.0] = d_max
    np.clip(depth_m, d_min, d_max, out=depth_m)
    depth_norm = (depth_m - d_min) / (d_max - d_min)
    out = _resize_like_sim(depth_norm)
    # Contract: the policy's conv weights are float32. Assert it rather than trust it.
    return out.astype(np.float32, copy=False)


def _side_capture_time(cam_data) -> float:
    """True capture time of the newest side frame, as host epoch seconds.

    Uses ``camera_capture_timestamp`` — librealsense's per-frame stamp, which with
    global time enabled (default on D400) is the sensor capture time mapped onto
    the host clock. Validated against the host receive time: if global time is
    off, the stamp is in the camera's hardware-clock domain (device uptime) and
    lands far from receive_time, in which case we fall back to receive_time and
    warn once.
    """
    recv = float(cam_data['timestamp'][-1])
    cap_arr = cam_data.get('camera_capture_timestamp')
    if cap_arr is not None:
        cap = float(cap_arr[-1])
        if abs(cap - recv) < 1.0:
            return cap
    if not getattr(_side_capture_time, '_warned', False):
        _side_capture_time._warned = True
        print("[WARN] RealSense camera_capture_timestamp is not in the host clock "
              "domain (global time disabled?); side latency falls back to SDK "
              "receive time and will understate the true frame age.")
    return recv


def _build_depth_obs(env, orbbec_bg, side_depth_scale):
    """Current (side_norm, front_norm, front_rgb, side_ts, front_ts) — the policy's depth inputs.

    Shared by the warmup and the control loop so both provably exercise the same
    dtype/shape path; the warmup previously used zeros and so caught neither.

    side_ts / front_ts are best-effort TRUE capture times on the host clock:
    the RealSense global-time frame stamp and the Orbbec global-timestamp-fitter
    stamp. Each degrades to its SDK receive time (with a one-time warning /
    source log) when the device stamp isn't host-clock comparable.
    """
    if env.last_realsense_data is None:
        raise RuntimeError("No realsense data yet — was RealEnv started with enable_depth=True?")
    rs_side = env.last_realsense_data[SIDE_CAM_IDX].get('depth')
    if rs_side is None:
        raise RuntimeError("Depth frames missing from realsense buffer.")
    side_ts = _side_capture_time(env.last_realsense_data[SIDE_CAM_IDX])

    frame = orbbec_bg.get_latest()
    if frame is None:
        raise RuntimeError("Orbbec produced no frame — front_depth unavailable.")
    front_rgb, front_depth_u16, front_ts = frame

    side_norm  = _process_u16_depth(rs_side[-1], side_depth_scale)
    front_norm = _process_u16_depth(front_depth_u16, ORBBEC_DEPTH_SCALE)
    return side_norm, front_norm, front_rgb, side_ts, front_ts


def _depth_to_bgr(depth_norm: np.ndarray) -> np.ndarray:
    """float32 [0,1] → BGR uint8 via TURBO colormap (matches demo_real_robot.py)."""
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


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


def _build_joint_pos(arm_joint_pos: np.ndarray, gripper_pos_raw: float,
                     include_gripper: bool = True) -> np.ndarray:
    """Reconstruct the 12-dim sim joint_pos from real-robot scalars.

    Args:
        arm_joint_pos: (6,) UR5e joint angles (rad).
        gripper_pos_raw: normalized gripper position in [0, 1] (0=open, 1=closed),
            as returned by RTDEInterpolationController using calibrated open/close positions.
        include_gripper: when False (--proprio_mode slim), skip the 6 synthetic
            mimic gripper joints and return only the (6,) arm joints.
    Returns: (12,) float32 in the order Isaac Lab returns
        ``asset.data.joint_pos`` for EXPLICIT_UR5E_ROBOTIQ_2F85
        (arm joints first, then 6 gripper joints driven by mimic).
    """
    if not include_gripper:
        return arm_joint_pos.astype(np.float32)
    master_angle = (float(gripper_pos_raw) - GRIPPER_POS_OPEN) * GRIPPER_POS_TO_RAD
    gripper_joints = (GRIPPER_MIMIC_RATIOS * master_angle).astype(np.float32)
    return np.concatenate([arm_joint_pos.astype(np.float32), gripper_joints], axis=0)


def _gripper_close_progress(gripper_pos_raw: float) -> np.ndarray:
    """Measured gripper close fraction (0=open, 1=closed) as a (1,) float32.

    Real counterpart of sim's ``task_mdp.gripper_close_progress`` (the extra
    proprio term the ...-ArmProprio-GripperAux-v0 task appends): sim normalises
    the Robotiq ``finger_joint`` angle over its command range
    ``(pos - open_pos) / (close_pos - open_pos)`` and ``clamp(0, 1)``.

    ``obs['gripper_pos']`` (``gripper_pos_raw``) is ALREADY that fraction, using
    the *calibrated* open/close values: the RTDE controller fetches them from the
    gripper API after activation (``gripper.get_open_position()`` /
    ``get_closed_position()``) and normalises the measured position as
    ``(raw - open) / (close - open)`` — see rtde_interpolation_controller.py:326
    and :385. So here we only clamp to [0, 1], matching sim's ``clamp``. Do NOT
    re-normalise: the calibration is already applied upstream, and doing it again
    would double-count it.
    """
    return np.array([np.clip(float(gripper_pos_raw), 0.0, 1.0)], dtype=np.float32)


def _build_proprio_tensor(history, device, include_prev_action: bool = True,
                          include_gripper_progress: bool = False) -> torch.Tensor:
    """Stack the per-frame history into the policy's proprio input.

    Args:
        history: deque of dicts {prev_action: (7,), joint_pos: (12,), ee_pose: (6,),
            gripper_progress: (1,)}, oldest-first. Pads short histories by
            repeating the earliest frame (matches Isaac Lab's CircularBuffer,
            which back-fills with the first observation seen at startup).
        include_prev_action: when False (--proprio_mode slim / slim_gripper), the
            prev_action term is omitted entirely; the remaining terms keep order.
        include_gripper_progress: when True (--proprio_mode slim_gripper), append
            the measured gripper close fraction term after ee_pose, matching sim's
            ProprioArmGripperProgressCfg (gripper_close_progress last).

    Returns: (1, 5*7 + 5*12 + 5*6) = (1, 125) tensor on ``device``
    (full mode; slim is (1, 5*6 + 5*6) = (1, 60); slim_gripper is
    (1, 5*6 + 5*6 + 5*1) = (1, 65)).
    Layout (matches Isaac Lab ObservationManager with concatenate_terms=True
    and flatten_history_dim=True): all 5 prev_action frames flat, then all 5
    joint_pos frames flat, then all 5 ee_pose frames flat, then (slim_gripper)
    all 5 gripper_progress frames flat.
    """
    if len(history) == 0:
        raise ValueError("Empty proprio history")
    while len(history) < HISTORY_LEN:
        history.appendleft(history[0])
    terms = []
    if include_prev_action:
        terms.append(np.concatenate([h["prev_action"] for h in history], axis=0))
    terms.append(np.concatenate([h["joint_pos"] for h in history], axis=0))
    terms.append(np.concatenate([h["ee_pose"]   for h in history], axis=0))
    if include_gripper_progress:
        terms.append(np.concatenate([h["gripper_progress"] for h in history], axis=0))
    flat = np.concatenate(terms, axis=0).astype(np.float32)
    return torch.from_numpy(flat).unsqueeze(0).to(device)


def _stack_view(history: deque, frame_t: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Append one single-frame view tensor (1, C, H, W) to its history and return the policy's
    vision input for that view.

    ``n_frames <= 1``  -> the frame itself (1, C, H, W), back-compatible with single-frame JITs.
    ``n_frames > 1``   -> (1, n_frames, C, H, W), ordered OLDEST-first / NEWEST-last, left-padded
    with the FIRST frame after reset until the window fills. This reproduces IsaacLab's
    ``CircularBuffer`` history padding (isaaclab/utils/buffers/circular_buffer.py:136-141): the
    first observation after a reset fills every history slot, and older slots keep that first
    frame until real frames overwrite them (newest -> oldest). E.g. for n_frames=4 with frames
    A,B,C,D,E: [A,A,A,A] -> [A,A,A,B] -> [A,A,B,C] -> [A,B,C,D] -> [B,C,D,E].

    ``history`` is a ``deque(maxlen=n_frames)`` and MUST be ``.clear()``-ed on every episode
    reset (alongside proprio_history) so a new episode re-pads from its own first frame.
    """
    history.append(frame_t)
    if n_frames <= 1:
        return frame_t
    frames = list(history)                    # oldest -> newest (append adds at the right)
    while len(frames) < n_frames:
        frames.insert(0, frames[0])           # left-pad with the first/oldest frame
    return torch.stack(frames, dim=1)         # (1, n_frames, C, H, W)


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
@click.option('--torch_device', default='cuda', type=str,
              help='Torch device for JIT inference.')
@click.option('--proprio_mode', default='full',
              type=click.Choice(['full', 'slim', 'slim_gripper']),
              help="Proprio layout: 'full' = prev_actions + joint_pos(12) + "
                   "ee_pose (125 dims); 'slim' drops prev_actions and the 6 "
                   "synthetic mimic gripper joints (60 dims), same term order; "
                   "'slim_gripper' is 'slim' plus a 1-d measured gripper close "
                   "fraction appended after ee_pose (65 dims), matching the "
                   "...-ArmProprio-GripperAux-v0 task.")
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints, max_duration,
         frequency, save_video, action_noise,
         collect_sysid, torch_device, proprio_mode):
    # slim / slim_gripper proprio: no prev_action term, arm-only joint_pos (see
    # _build_proprio_tensor). slim_gripper additionally appends the measured
    # gripper close fraction term after ee_pose.
    include_prev_action = proprio_mode == 'full'
    include_gripper_joints = proprio_mode == 'full'
    include_gripper_progress = proprio_mode == 'slim_gripper'
    print(f"Proprio mode: {proprio_mode}")
    # Per-axis Cartesian scale matching simulation DiffIK config.
    # Identical to eval_real_robot.py — sim's RelCartesianOSCEvalAction scales.
    CARTESIAN_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")
    print(f"Depth clip: {DEPTH_CLIP} m (no-return → {DEPTH_CLIP[1]} m)")

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
        # Imported lazily: skvideo is absent from the foundstereo env this script
        # now runs in, and --match_dataset is an optional parity feature.
        import skvideo.io
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
    # Only the D435 side camera goes through RealEnv; front comes from the Orbbec.
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
    ]

    device = torch.device(torch_device if torch.cuda.is_available() and torch_device.startswith('cuda') else 'cpu')
    print(f"Loading depth policy JIT from {input} on {device}")
    policy = torch.jit.load(input, map_location=device)
    policy.eval()
    meta = _load_jit_metadata(input)
    expected_proprio = int(meta.get("num_proprio", 0))
    expected_h = int(meta.get("image_h", DEPTH_IMG_H))
    expected_w = int(meta.get("image_w", DEPTH_IMG_W))
    expected_groups = meta.get("vision_groups", "side_depth,front_depth").split(",")
    if expected_groups != ["side_depth", "front_depth"]:
        raise ValueError(
            f"This eval script is wired for vision_groups=['side_depth', 'front_depth']; "
            f"JIT was trained with {expected_groups}. Update the camera plumbing if "
            f"the policy expects a different camera set."
        )
    if (expected_h, expected_w) != (DEPTH_IMG_H, DEPTH_IMG_W):
        raise ValueError(
            f"JIT expects {expected_h}x{expected_w} depth, but this script always "
            f"resizes to {DEPTH_IMG_H}x{DEPTH_IMG_W}. Update DEPTH_IMG_H/W to match."
        )
    # Frame-stacked policies (e.g. the 4-frame depth students) declare n_frames>1 in the sidecar.
    # We then keep a per-view depth-frame history and feed (1, n_frames, 1, H, W) per view, padded
    # per IsaacLab CircularBuffer semantics (see _stack_view). n_frames==1 keeps the single-frame path.
    expected_n_frames = int(meta.get("n_frames", 1))
    if expected_n_frames > 1:
        print(f"[frame stacking] policy expects n_frames={expected_n_frames} per depth view; "
              f"maintaining per-view image history with first-frame (IsaacLab) padding.")

    # ── setup experiments ──────────────────────────────────────────────────
    dt = 1 / frequency
    # Need at least HISTORY_LEN obs frames per call so RealEnv's deque
    # carries enough context for the proprio history we build.
    n_obs_steps = HISTORY_LEN
    print(f"n_obs_steps (matches HISTORY_LEN): {n_obs_steps}")
    print("Policy outputs single-step actions (n_action_steps=1)")

    side_depth_mean = 0.0
    side_depth_std = 0.0
    front_depth_mean = 0.0
    front_depth_std = 0.0
    timestep = 0
    # Per-step frame-age/skew samples (seconds), measured right before inference.
    # age = now - true capture time (device stamp mapped to host clock, see
    # _build_depth_obs); skew = side_ts - front_ts (positive → side frame newer).
    side_age_log, front_age_log, cam_skew_log = [], [], []

    def _print_depth_stats():
        """Running mean/std of the normalised policy inputs — useful for spotting
        a sim/real gap in the depth distribution. No-ops before the first step."""
        if timestep == 0:
            print("No depth stats yet (0 steps).")
            return
        print(f"Side depth  mean: {side_depth_mean / timestep:.4f}, "
              f"std: {side_depth_std / timestep:.4f}")
        print(f"Front depth mean: {front_depth_mean / timestep:.4f}, "
              f"std: {front_depth_std / timestep:.4f}")
        _print_latency_stats()

    def _print_latency_stats():
        """Frame age at inference time + side-vs-front skew, in ms."""
        if not side_age_log:
            print("No latency stats yet (0 steps).")
            return
        for name, log in (("side age ", side_age_log),
                          ("front age", front_age_log),
                          ("skew s-f ", cam_skew_log)):
            a = np.asarray(log) * 1000.0
            print(f"[latency] {name}: mean {a.mean():6.1f} ms, std {a.std():5.1f}, "
                  f"p50 {np.percentile(a, 50):6.1f}, p95 {np.percentile(a, 95):6.1f}, "
                  f"max {a.max():6.1f}  (n={len(a)})")

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shm_manager:
        # The Orbbec joins the `with` so its pipeline is always torn down, even if
        # startup (policy warmup, RealEnv obs) raises — otherwise the SDK aborts
        # the process and hides the real traceback.
        with OrbbecDepthBackground() as orbbec_bg, \
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
                camera_serial_numbers=[SIDE_SERIAL],
                camera_configs=configs,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)

            print("Waiting for realsense")
            time.sleep(5.0)

            # Depth scale — read once after the camera is ready.
            side_depth_scale = env.realsense.cameras[SIDE_SERIAL].get_depth_scale()
            print(f'Depth scale — side: {side_depth_scale:.5f} m/unit')

            _preflight_realsense(env, n_obs_steps)

            # ── Depth video writer (mirrors demo_real_robot.py panels) ─────
            depth_video_writer = None
            depth_raw_video_writer = None
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

            # ── Warm up policy with current obs (no inference latency in step 0) ──
            print("Warming up policy inference")
            obs = env.get_obs()
            obs['end_effector_pose'] = compute_calibrated_ee_pose(obs['arm_joint_pos'])
            arm_jp_now = obs['arm_joint_pos'][-1]
            grip_now = _last_scalar(obs['gripper_pos'])
            warmup_frame = {
                "prev_action": np.zeros(PREV_ACTION_DIM, dtype=np.float32),
                "joint_pos":   _build_joint_pos(arm_jp_now, grip_now,
                                                include_gripper=include_gripper_joints),
                "ee_pose":     obs['end_effector_pose'][-1].astype(np.float32),
                "gripper_progress": _gripper_close_progress(grip_now),
            }
            warmup_history = deque([warmup_frame] * HISTORY_LEN, maxlen=HISTORY_LEN)
            warmup_proprio = _build_proprio_tensor(
                warmup_history, device, include_prev_action=include_prev_action,
                include_gripper_progress=include_gripper_progress)
            if expected_proprio and warmup_proprio.shape[-1] != expected_proprio:
                raise ValueError(
                    f"proprio dim mismatch: built {warmup_proprio.shape[-1]} but JIT "
                    f"expects {expected_proprio}. Check --proprio_mode "
                    f"({proprio_mode}), joint_pos construction "
                    f"(NUM_JOINTS={NUM_JOINTS}) and the proprio cfg."
                )
            # Warm up on REAL depth, through the same helpers the loop uses, rather
            # than on zeros: a zeros warmup exercises neither the true dtype nor the
            # true shape, so a bad frame would only surface at iteration 0 of a live
            # episode — with the robot already moving. Fail here instead.
            warmup_side, warmup_front = _build_depth_obs(
                env, orbbec_bg, side_depth_scale)[:2]
            warmup_side_t  = torch.from_numpy(warmup_side).to(device, torch.float32)[None, None]
            warmup_front_t = torch.from_numpy(warmup_front).to(device, torch.float32)[None, None]
            # Exercise the exact (frame-stacked) vision shape the loop will feed, using throwaway
            # per-view histories so a shape/dtype bug fails here rather than mid-episode.
            _ws_hist = deque(maxlen=expected_n_frames)
            _wf_hist = deque(maxlen=expected_n_frames)
            warmup_side_in  = _stack_view(_ws_hist, warmup_side_t, expected_n_frames)
            warmup_front_in = _stack_view(_wf_hist, warmup_front_t, expected_n_frames)
            with torch.no_grad():
                action_mean = policy(warmup_proprio, [warmup_side_in, warmup_front_in])
                print(f"Warmup OK on live depth "
                      f"(side {warmup_side.dtype}{warmup_side.shape}, "
                      f"front {warmup_front.dtype}{warmup_front.shape}); "
                      f"action shape={tuple(action_mean.shape)}")
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

            # Per-episode state — reset on each new episode start.
            proprio_history: deque = deque(maxlen=HISTORY_LEN)
            # Per-view depth-frame history for frame-stacked policies (n_frames>1). Cleared on
            # every episode reset alongside proprio_history; _stack_view left-pads with the first
            # post-reset frame (IsaacLab CircularBuffer semantics). maxlen=1 is a no-op passthrough.
            side_img_history: deque = deque(maxlen=expected_n_frames)
            front_img_history: deque = deque(maxlen=expected_n_frames)

            while True:
                # ========== policy control loop ==============
                try:
                    proprio_history.clear()
                    side_img_history.clear()
                    front_img_history.clear()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start, time_func=time.time)
                    print("Started!")
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

                        # ── Build depth obs (same helper the warmup used) ──
                        side_norm, front_norm, front_rgb_raw, side_ts, front_ts = \
                            _build_depth_obs(env, orbbec_bg, side_depth_scale)

                        # Capture concatenated RGB frames if recording
                        if save_video:
                            imgs = []
                            for img in (front_rgb_raw, obs['side_rgb'][-1]):
                                if img.dtype in (np.float32, np.float64):
                                    img = (img * 255).clip(0, 255).astype(np.uint8)
                                imgs.append(img)
                            # Orbbec is 1280x960, D435 obs is 640x480 — match heights
                            # before concatenating.
                            h = min(i.shape[0] for i in imgs)
                            imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h))
                                    for i in imgs]
                            frame = np.concatenate(imgs, axis=1)
                            if episode_video_writer is not None:
                                episode_video_writer.append_data(frame)
                            if long_video_writer is not None:
                                long_video_writer.append_data(frame)

                        # Visualise depth panels (mirrors demo_real_robot.py overlays)
                        side_vis  = _depth_to_bgr(side_norm)
                        front_vis = _depth_to_bgr(front_norm)
                        side_color  = cv2.resize(obs['side_rgb'][-1][:, :, ::-1], (DEPTH_IMG_W, DEPTH_IMG_H))
                        front_color = cv2.resize(front_rgb_raw[:, :, ::-1],       (DEPTH_IMG_W, DEPTH_IMG_H))
                        if side_color.dtype != np.uint8:
                            side_color = (np.clip(side_color, 0.0, 1.0) * 255).astype(np.uint8)
                        if front_color.dtype != np.uint8:
                            front_color = (np.clip(front_color, 0.0, 1.0) * 255).astype(np.uint8)
                        side_overlay  = cv2.addWeighted(side_vis,  0.5, side_color,  0.5, 0)
                        front_overlay = cv2.addWeighted(front_vis, 0.5, front_color, 0.5, 0)
                        cv2.putText(side_overlay,  'SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        cv2.putText(front_overlay, 'FRONT', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        depth_panel = np.concatenate([side_overlay, front_overlay], axis=1)
                        if depth_video_writer is not None:
                            depth_video_writer.write(depth_panel)
                        if depth_raw_video_writer is not None:
                            side_vis_raw  = _depth_to_bgr(side_norm).copy()
                            front_vis_raw = _depth_to_bgr(front_norm).copy()
                            cv2.putText(side_vis_raw,  'SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            cv2.putText(front_vis_raw, 'FRONT', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                            depth_raw_panel = np.concatenate([side_vis_raw, front_vis_raw], axis=1)
                            depth_raw_video_writer.write(depth_raw_panel)

                        # ── Build proprio frame and append to history ──
                        arm_jp_now = obs['arm_joint_pos'][-1]
                        grip_pos_raw = _last_scalar(obs['gripper_pos'])
                        ee_pose_now  = obs['end_effector_pose'][-1].astype(np.float32)
                        proprio_history.append({
                            # prev_action = the raw action RealEnv last executed (single source
                            # of truth; tracked in exec_actions, never stalls, zeroed per episode).
                            "prev_action": env.get_last_action(),
                            "joint_pos":   _build_joint_pos(arm_jp_now, grip_pos_raw,
                                                            include_gripper=include_gripper_joints),
                            "ee_pose":     ee_pose_now,
                            "gripper_progress": _gripper_close_progress(grip_pos_raw),
                        })
                        proprio_tensor = _build_proprio_tensor(
                            proprio_history, device, include_prev_action=include_prev_action,
                            include_gripper_progress=include_gripper_progress)
                        # dtype is pinned explicitly: the policy's conv weights are
                        # float32, and a float64 input fails deep inside the JIT.
                        side_t  = torch.from_numpy(side_norm).to(device, torch.float32)[None, None]   # (1,1,H,W)
                        front_t = torch.from_numpy(front_norm).to(device, torch.float32)[None, None]

                        side_depth_mean += side_norm.mean()
                        side_depth_std += side_norm.std()
                        front_depth_mean += front_norm.mean()
                        front_depth_std += front_norm.std()
                        timestep += 1
                        # Frame age at the moment of inference (includes video/vis
                        # processing above), plus side-vs-front capture skew.
                        _now = time.time()
                        side_age_log.append(_now - side_ts)
                        front_age_log.append(_now - front_ts)
                        cam_skew_log.append(side_ts - front_ts)
                        if timestep % 100 == 0:
                            _print_latency_stats()
                        # ── Run inference ──
                        # Frame-stack: append this frame to each view's history and left-pad with
                        # the first post-reset frame (IsaacLab CircularBuffer). n_frames==1 -> the
                        # single (1,1,H,W) frame unchanged. List order must match vision_groups =
                        # [side_depth, front_depth].
                        side_in  = _stack_view(side_img_history, side_t, expected_n_frames)
                        front_in = _stack_view(front_img_history, front_t, expected_n_frames)
                        with torch.no_grad():
                            action_mean = policy(proprio_tensor, [side_in, front_in]).cpu().numpy()
                        # action_mean: (1, num_actions=7) — [arm_delta(6), gripper(1)]
                        raw_action = action_mean[0]  # (7,)
                        if action_noise > 0:
                            raw_action[:6] = raw_action[:6] + np.random.randn(6) * action_noise

                        # Gripper open macro
                        gripper_action = raw_action[6:7].copy()
                        if gripper_open_steps_remaining > 0:
                            gripper_action = np.array([1.0], dtype=np.float32)
                            gripper_open_steps_remaining -= 1
                            if gripper_open_steps_remaining == 0:
                                print("[Gripper macro] done, returning to policy control")
                        # prev_action is now tracked by RealEnv (set from obs_actions in
                        # exec_actions below); no local last_raw_action bookkeeping needed.

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
                            _print_depth_stats()
                            break
                        elif key_stroke == ord('r'):
                            save_sysid_data()
                            sysid_records.clear()
                            print('Resetting robot for new trajectory...')
                            env.end_episode()
                            if save_video and episode_video_writer is not None:
                                episode_video_writer.close()
                                episode_video_writer = None
                                print(f"  Episode video saved.")
                            proprio_history.clear()
                            side_img_history.clear()
                            front_img_history.clear()
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
                            front_depth_mean = 0.0
                            front_depth_std = 0.0
                            side_age_log.clear()
                            front_age_log.clear()
                            cam_skew_log.clear()
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
                    # Orbbec teardown is handled by the `with` block's __exit__.
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
                # NOTE: no orbbec_bg.stop() here — this is the tail of the outer
                # `while True`, which loops round to start another episode. Only
                # the except branch truly exits, so teardown lives there.


# %%
if __name__ == '__main__':
    main()
