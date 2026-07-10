"""Real-robot inference wrapper for a BC PointNet (see POINTCLOUD_EVAL.md).

Hardware twin of the sim path ``bc_utils.bc_actions`` + the sim eval config
``Ur5eRobotiq2f85BCPointNetSegEvalCfg``. Wraps the vendored ``load_bc_pointnet`` and
re-implements ``bc_actions`` on raw tensors (no Isaac obs dict), plus the 18-d proprio
reconstruction from real-robot scalars.

Observation layout is driven by the checkpoint's ``pc_signature`` (UWLab pc_signature.py;
in the JIT ``.meta.json`` or the Lightning hparams) when present -- cloud classes /
per-class budget / seg channel / frame plus the proprio term layout -- via
:func:`perception_from_signature` and :meth:`PointNetPolicy.proprio_layout`. Checkpoints
that predate the signature fall back to the original conventions:
  * cloud   : (num_points, 4) = xyz + seg label {robot:0, peg:-1, hole:+1}, in EE frame
  * proprio : by proprio_dim -- 18-d = [joint_pos(12: 6 arm + 6 Robotiq mimic, rad),
              ee_pose(6: xyz + axis-angle, wrist_3_link in BASE frame)]; 12-d drops the 6
              mimic gripper joints; 6-d is the arm joints alone. Declaration order, NO
              prev_actions.
  * action  : 7-d = RelCartesian OSC dpose(6) + binary gripper(1), denormalized
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from diffusion_policy.model.point_cloud import load_bc_pointnet
from diffusion_policy.real_world.pointcloud_builder import SEG_LABELS
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle

# Sim class names (pc_signature "classes") -> the real pipeline's SAM2/SEG_LABELS names.
SIM2REAL_CLASS = {"robot": "robot", "insertive": "peg", "receptive": "hole"}


def perception_from_signature(sig, policy):
    """Perception config from the checkpoint's PC observation signature (UWLab
    ``pc_signature.py``, carried in the JIT ``.meta.json`` / Lightning hparams).

    Returns ``(budget, prompt_classes, label_remap)``:
      * budget: {real seg label -> point count} for ``RealEnv.setup_pointcloud`` -- the
        sim's exact per-class split (largest-remainder counts saved in the signature).
      * prompt_classes: real class names to SAM2-prompt (classes with a non-zero budget);
        e.g. an occluded-objects policy (robot ratio 0) skips the robot click + tracking.
      * label_remap: {real label value -> trained label value} for the cloud's seg channel,
        or None when the trained labels already match ``SEG_LABELS`` (the default).
    ``(None, None, None)`` when the checkpoint has no signature (old export) -- caller
    falls back to the legacy defaults.
    """
    if not sig:
        print("[signature] checkpoint has NO pc_signature -- falling back to legacy defaults "
              "(DEFAULT_BUDGET, prompt robot/peg/hole). Re-export from a run/dataset that has one.")
        return None, None, None
    if sig.get("pc_parts"):
        raise NotImplementedError(
            f"pc_signature has per-prim parts {sig['pc_parts']}; the real pipeline segments "
            f"whole classes (robot/peg/hole) and cannot reproduce per-prim selection yet.")
    frame = sig.get("frame")
    if frame != "wrist_3_link":
        raise ValueError(
            f"policy cloud frame is '{frame}' but the real pipeline builds wrist_3_link (EE) "
            f"frame clouds only; this checkpoint cannot be evaluated with this script.")
    counts = sig.get("class_points")
    if not counts:
        print("[signature] no class_points in signature (legacy fixed split); using DEFAULT_BUDGET.")
        return None, None, None
    budget, prompt_classes = {}, []
    for sim_name in sig["classes"]:
        n = int(counts.get(sim_name, 0))
        real_name = SIM2REAL_CLASS[sim_name]
        if n > 0:
            budget[SEG_LABELS[real_name]] = n
            prompt_classes.append(real_name)
    # Trained seg-channel values (sim names) vs what the real builder emits (SEG_LABELS).
    label_remap = None
    if sig.get("include_segmentation") or sig.get("append_prim_semantic"):
        trained = sig.get("segmentation_labels") or {}
        remap = {}
        for sim_name, real_name in SIM2REAL_CLASS.items():
            if sim_name in trained and float(trained[sim_name]) != float(SEG_LABELS[real_name]):
                remap[float(SEG_LABELS[real_name])] = float(trained[sim_name])
        label_remap = remap or None
    total = sum(budget.values())
    if policy.num_points and total != int(policy.num_points):
        print(f"[signature] WARNING: budget sums to {total} but the model trained on "
              f"{policy.num_points} points.")
    print(f"[signature] cloud: {sig['pc_term']} | frame={frame} | {total} pts = "
          + ", ".join(f"{c}:{budget.get(SEG_LABELS[SIM2REAL_CLASS[c]], 0)}" for c in sig["classes"])
          + f" | point_dim={sig['point_dim']} (seg={'on' if sig['point_dim'] == 4 else 'off'})"
          + (f" | label remap {label_remap}" if label_remap else ""))
    if sig.get("robot_body_names"):
        print(f"[signature] NOTE: sim robot cloud was FILTERED to bodies {sig['robot_body_names']} "
              f"-- prompt SAM2 'robot' on those parts only (e.g. gripper, not the whole arm).")
    print(f"[signature] proprio: {sig['proprio']}")
    return budget, prompt_classes, label_remap

# ── Robotiq 2F-85 gripper joint reconstruction (mirrors eval_real_robot_depth.py) ──
# Real hardware reports one normalized gripper position in [0, 1] (0=open, 1=closed).
# Sim's joint_pos has 6 gripper joints driven by the finger_joint master via mimic
# constraints; reconstruct them so the 12-d joint_pos matches what the model trained on.
GRIPPER_POS_OPEN = 0.0
GRIPPER_POS_CLOSE = 1.0
GRIPPER_POS_TO_RAD = np.pi / 4 / (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN)
# Per-column sign of the 6 gripper joints w.r.t. the master, in Isaac Lab's
# ``robot.data.joint_pos`` order for EXPLICIT_UR5E_ROBOTIQ_2F85.
GRIPPER_MIMIC_RATIOS = np.array([+1.0, +1.0, -1.0, +1.0, -1.0, -1.0], dtype=np.float32)


def build_joint_pos(arm_joint_pos: np.ndarray, gripper_pos_raw: float) -> np.ndarray:
    """(6,) arm joints + scalar gripper pos -> (12,) sim joint_pos (arm, then 6 mimic)."""
    master = (float(gripper_pos_raw) - GRIPPER_POS_OPEN) * GRIPPER_POS_TO_RAD
    gripper_joints = GRIPPER_MIMIC_RATIOS * master
    return np.concatenate([np.asarray(arm_joint_pos, np.float32), gripper_joints]).astype(np.float32)


def build_ee_pose(arm_joint_pos: np.ndarray) -> np.ndarray:
    """(6,) arm joints -> (6,) [x,y,z, rx,ry,rz] EE pose (wrist_3_link in base, axis-angle)."""
    pos, quat = get_ee_pose(np.asarray(arm_joint_pos, np.float64))
    return np.concatenate([pos, quat_to_axis_angle(quat)]).astype(np.float32)


def build_proprio(arm_joint_pos: np.ndarray, gripper_pos_raw: float,
                  include_gripper_joints: bool = True,
                  include_ee_pose: bool = True) -> np.ndarray:
    """Assemble the proprio vector in the trained declaration order.

    ``include_gripper_joints=True``  -> joint_pos(12: 6 arm + 6 Robotiq mimic reconstructed
    from the scalar gripper position); ``False`` -> arm_joint_pos(6) only (real-robot
    convention: the mimic gripper joints don't exist on hardware). ``include_ee_pose``
    appends ee_pose(6: xyz + axis-angle, wrist_3_link in base) -- drop for models whose
    proprio is joint positions only. So: 18-d = 12+6, 12-d = 6+6, 6-d = arm joints alone.
    """
    joint_pos = (build_joint_pos(arm_joint_pos, gripper_pos_raw) if include_gripper_joints
                 else np.asarray(arm_joint_pos, np.float32))
    if not include_ee_pose:
        return np.asarray(joint_pos, np.float32)
    return np.concatenate([joint_pos, build_ee_pose(arm_joint_pos)]).astype(np.float32)


def _looks_like_jit(path: str) -> bool:
    """Heuristic: a JIT export from ``convert_bc_to_jit.py`` has a ``<path>.meta.json`` sidecar
    (and is typically a ``.pt``); an eager Lightning policy is a ``.ckpt``."""
    return os.path.exists(path + ".meta.json") or (
        path.endswith(".pt") and not path.endswith(".ckpt"))


class PointNetPolicy:
    """BC PointNet policy: (cloud, proprio) -> denormalized action.

    Supports two checkpoint formats with one API:
      * eager  -- a Lightning ``.ckpt`` via ``load_bc_pointnet``; proprio z-scoring and action
                  de-normalization are applied here from the checkpoint's saved stats.
      * jit    -- a traced module from ``convert_bc_to_jit.py`` whose ``forward(points, proprio)``
                  has both normalizations BAKED IN (+ a ``<path>.meta.json`` sidecar for dims).

    ``jit=None`` (default) auto-detects via the sidecar / extension; pass ``True``/``False`` to
    force. Either way ``predict`` / ``predict_from_state`` return the denormalized env action.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda", jit: bool = None):
        self.device = device
        self.jit = _looks_like_jit(ckpt_path) if jit is None else bool(jit)
        if self.jit:
            self.bc = None
            self.model = torch.jit.load(ckpt_path, map_location=device).eval()
            meta = {}
            if os.path.exists(ckpt_path + ".meta.json"):
                with open(ckpt_path + ".meta.json") as f:
                    meta = json.load(f)
            self.point_dim = int(meta.get("point_dim", 4))
            self.num_points = meta.get("num_points")
            self.proprio_dim = int(meta.get("proprio_dim", 18))
            self.action_dim = int(meta.get("action_dim", 7))
            # PC observation signature (UWLab pc_signature.py): how the training cloud +
            # proprio were built (classes, per-class budget, frame, seg labels, proprio
            # term layout). None for exports that predate it.
            self.pc_signature = meta.get("pc_signature")
        else:
            self.bc = load_bc_pointnet(ckpt_path, device)
            self.model = self.bc["model"]
            hp = self.bc["hp"]
            self.point_dim = self.model.point_dim
            self.num_points = hp.get("num_points")
            self.proprio_dim = int(self.bc["proprio_mean"].shape[0])
            self.action_dim = int(self.bc["action_mean"].shape[0])
            self.pc_signature = hp.get("pc_signature")
        sig_dim = (self.pc_signature or {}).get("proprio", {}).get("dim")
        if sig_dim is not None and int(sig_dim) != self.proprio_dim:
            raise ValueError(
                f"pc_signature proprio dim {sig_dim} != checkpoint proprio_dim {self.proprio_dim}; "
                f"the signature does not describe this checkpoint")

    @torch.no_grad()
    def predict(self, points, proprio) -> np.ndarray:
        """Run the policy. Returns the denormalized action in env units.

        Args:
            points:  (N, point_dim) or (B, N, point_dim) -- cloud, 4th channel = seg label.
            proprio: (proprio_dim,) or (B, proprio_dim) -- raw; z-scoring is applied (eager) or
                     baked into the traced graph (jit).
        Returns:
            (action_dim,) if a single sample was given, else (B, action_dim).
        """
        pts = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=self.device)
        pr = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=self.device)
        squeeze = pts.ndim == 2
        if squeeze:
            pts = pts.unsqueeze(0)
        if pr.ndim == 1:
            pr = pr.unsqueeze(0)
        assert pts.shape[-1] == self.point_dim, f"cloud channel {pts.shape[-1]} != point_dim {self.point_dim}"
        assert pr.shape[-1] == self.proprio_dim, f"proprio dim {pr.shape[-1]} != {self.proprio_dim}"

        if self.jit:
            action = self.model(pts, pr)  # forward bakes in proprio z-score + action denorm
        else:
            pr_n = (pr - self.bc["proprio_mean"]) / self.bc["proprio_std"]
            out = self.model(pts, pr_n)
            mean = out[0] if isinstance(out, tuple) else out  # predict_std -> (mean, log_std)
            action = mean * self.bc["action_std"] + self.bc["action_mean"]
        action = action.cpu().numpy()
        return action[0] if squeeze else action

    def proprio_layout(self) -> tuple:
        """``(include_gripper_joints, include_ee_pose)`` for :func:`build_proprio`.

        Prefers the checkpoint's ``pc_signature`` (explicit term list + ``joint_pos_dims``);
        falls back to the ``proprio_dim`` heuristic for old checkpoints: 18 -> arm+gripper
        joints + ee_pose, 12 -> arm joints + ee_pose (no gripper), 6 -> arm joints only.
        """
        sig = self.pc_signature
        if sig and sig.get("proprio", {}).get("terms"):
            pr = sig["proprio"]
            jpd = pr.get("joint_pos_dims")
            include_gripper_joints = jpd is None or int(jpd) > 6
            include_ee_pose = "end_effector_pose" in pr["terms"]
            return include_gripper_joints, include_ee_pose
        if self.proprio_dim == 18:
            return True, True
        if self.proprio_dim == 12:
            return False, True
        if self.proprio_dim == 6:
            return False, False
        raise ValueError(
            f"unsupported proprio_dim {self.proprio_dim} and no pc_signature; expected 18 "
            f"(arm+gripper+ee), 12 (arm+ee, no gripper) or 6 (arm joints only)")

    def predict_from_state(self, points, arm_joint_pos, gripper_pos_raw) -> np.ndarray:
        """Convenience: assemble proprio from raw robot scalars (per :func:`proprio_layout`),
        then predict."""
        include_gripper_joints, include_ee_pose = self.proprio_layout()
        return self.predict(
            points,
            build_proprio(arm_joint_pos, gripper_pos_raw, include_gripper_joints, include_ee_pose))
