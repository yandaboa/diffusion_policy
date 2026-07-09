"""Real-robot inference wrapper for a BC PointNet (see POINTCLOUD_EVAL.md).

Hardware twin of the sim path ``bc_utils.bc_actions`` + the sim eval config
``Ur5eRobotiq2f85BCPointNetSegEvalCfg``. Wraps the vendored ``load_bc_pointnet`` and
re-implements ``bc_actions`` on raw tensors (no Isaac obs dict), plus the 18-d proprio
reconstruction from real-robot scalars.

Locked to ``pnocc_xl_residual_big_ee``:
  * cloud   : (num_points, 4) = xyz + seg label {robot:0, peg:-1, hole:+1}, in EE frame
  * proprio : 18-d = [joint_pos(12: 6 arm + 6 Robotiq mimic, rad), ee_pose(6: xyz +
              axis-angle, wrist_3_link in BASE frame)]  -- declaration order, NO prev_actions.
              ``*_no_gripper`` checkpoints use 12-d = [arm_joint_pos(6), ee_pose(6)] (the 6
              mimic gripper joints dropped); selected by the checkpoint's proprio_dim.
  * action  : 7-d = RelCartesian OSC dpose(6) + binary gripper(1), denormalized
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from diffusion_policy.model.point_cloud import load_bc_pointnet
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle

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
                  include_gripper_joints: bool = True) -> np.ndarray:
    """Assemble the proprio vector in the trained declaration order.

    ``include_gripper_joints=True``  -> 18-d = [joint_pos(12: 6 arm + 6 Robotiq mimic), ee_pose(6)].
    ``include_gripper_joints=False`` -> 12-d = [arm_joint_pos(6), ee_pose(6)]; the 6 made-up
    gripper mimic joints are dropped (the ``*_no_gripper`` models were trained without them).
    """
    joint_pos = (build_joint_pos(arm_joint_pos, gripper_pos_raw) if include_gripper_joints
                 else np.asarray(arm_joint_pos, np.float32))
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
        else:
            self.bc = load_bc_pointnet(ckpt_path, device)
            self.model = self.bc["model"]
            hp = self.bc["hp"]
            self.point_dim = self.model.point_dim
            self.num_points = hp.get("num_points")
            self.proprio_dim = int(self.bc["proprio_mean"].shape[0])
            self.action_dim = int(self.bc["action_mean"].shape[0])

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

    def predict_from_state(self, points, arm_joint_pos, gripper_pos_raw) -> np.ndarray:
        """Convenience: assemble proprio from raw robot scalars, then predict.

        Proprio layout follows the checkpoint's ``proprio_dim``: 18 -> arm+gripper joints+ee_pose;
        12 -> arm joints + ee_pose only (``*_no_gripper`` models drop the mimic gripper joints).
        """
        if self.proprio_dim == 18:
            include_gripper_joints = True
        elif self.proprio_dim == 12:
            include_gripper_joints = False
        else:
            raise ValueError(
                f"unsupported proprio_dim {self.proprio_dim}; expected 18 (arm+gripper+ee) "
                f"or 12 (arm+ee, no_gripper)")
        return self.predict(
            points, build_proprio(arm_joint_pos, gripper_pos_raw, include_gripper_joints))
