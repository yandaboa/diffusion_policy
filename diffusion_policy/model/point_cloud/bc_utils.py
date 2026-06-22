# VENDORED from UWLab-patrick-private/scripts/imitation_learning/point_cloud/bc_utils.py
# Verbatim copy — keep in sync with upstream if the BC PointNet model/checkpoint format changes.
# Deployed checkpoint: pnocc_xl_residual_big_ee (residual_point_net, point_dim=4, proprio_dim=18, action_dim=7).

# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-time helpers for a behavior-cloned :class:`PointNet`.

Reconstructs the raw :class:`PointNet` (+ the proprio/action z-scoring stats) from a
Lightning checkpoint written by ``train_point_net.py``, and runs it on a live
observation group -- so rolling out a BC PointNet (e.g. in ``play.py``) needs no
Lightning dependency, only ``torch`` + ``point_net.py``.
"""

from __future__ import annotations

import os
import sys

import torch

# Policy modules live next to this file; make them importable regardless of caller cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flatten_mlp import FlattenMLP  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

# Must match train_point_net.py's _ARCHITECTURES so a checkpoint loads into the class it trained.
_ARCHITECTURES = {"point_net": PointNet, "flatten_mlp": FlattenMLP, "residual_point_net": ResidualPointNet}

# Obs terms that are point clouds (stored flat as (N, num_points*point_dim)) -> the PointNet's set input.
# Mirrors train_point_net.py so eval proprio is built exactly as the model was trained.
_BC_PC_TERMS = {"scene_pc"}
# Proprio is an explicit ALLOWLIST mirroring train_point_net._PROPRIO_TERMS: the model sees
# only these terms (in the obs-group's declaration order) as the 18d proprio it was trained on.
# Any other term in the live group (privileged poses, contact flags, ... recorded for aux
# losses) is ignored at deploy, exactly as the trainer ignored it.
_BC_PROPRIO_TERMS = {"joint_pos", "end_effector_pose"}


def load_bc_pointnet(path: str, device: str) -> dict:
    """Reconstruct a PointNet (+ saved normalization stats) from a Lightning BC checkpoint.

    The checkpoint is the plain ``torch.save`` dict Lightning writes: ``hyper_parameters``
    holds the PointNet shape, ``state_dict`` holds ``model.*`` weights plus the
    proprio/action z-scoring buffers. We rebuild the raw :class:`PointNet` so eval needs
    no Lightning dependency.

    Returns a dict with ``model`` (eval-mode PointNet on ``device``), ``hp`` (the saved
    hyper-parameters), and the four normalization tensors (``proprio_mean/std``,
    ``action_mean/std``).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]
    arch = hp.get("architecture", "point_net")  # older ckpts predate the field -> PointNet
    kwargs = dict(
        encoder_hidden_dims=hp["encoder_dims"],
        action_hidden_dims=hp["action_dims"],
        proprio_dim=hp["proprio_dim"],
        action_dim=hp["action_dim"],
        predict_std=hp["predict_std"],
        point_dim=hp.get("point_dim", 3),  # 3 = xyz; 4 = xyz + segmentation label
    )
    if arch == "flatten_mlp":
        kwargs["num_points"] = hp["num_points"]  # FlattenMLP needs the fixed cloud size
    model = _ARCHITECTURES[arch](**kwargs)
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    model.to(device).eval()
    norm = {k: sd[k].to(device) for k in ("proprio_mean", "proprio_std", "action_mean", "action_std")}
    return {"model": model, "hp": hp, **norm}


def bc_actions(bc: dict, obs, group_name: str) -> torch.Tensor:
    """Run the BC PointNet on the ``group_name`` obs group and return denormalized actions.

    The group is a per-term dict (``concatenate_terms=False``): the point cloud term is
    reshaped to ``(N, num_points, point_dim)`` and the allowlisted proprio terms
    (``_BC_PROPRIO_TERMS``) are concatenated, in declaration order, into proprio, exactly as
    ``train_point_net.py`` did. Any other term in the group (privileged poses, contact flags
    recorded for aux losses) is ignored. Proprio is z-scored with the saved stats; the
    predicted (normalized) action is denormalized back to env action units.
    """
    g = obs[group_name]
    keys = list(g.keys())
    pc_key = next(k for k in keys if k in _BC_PC_TERMS)
    # point_dim (3 or 4) is baked into the model; the live scene_pc must match.
    pc = g[pc_key].reshape(g[pc_key].shape[0], -1, bc["model"].point_dim)
    proprio_keys = [k for k in keys if k in _BC_PROPRIO_TERMS]
    # Real-robot proprio trim: the model's proprio_dim (== saved proprio_mean length) may be SMALLER
    # than the live obs provides. That deficit is the trailing joint_pos entries -- the last 6 of the
    # 12 joint_pos dims are Robotiq gripper mimic joints that DO NOT EXIST on the real robot, so a
    # sim2real policy was trained on joint_pos[:6] (see train_point_net.py joint_pos_dims). Drop the
    # same trailing joint_pos dims here so eval/deploy proprio matches what the model trained on.
    target = bc["proprio_mean"].shape[0]
    deficit = sum(g[k].shape[-1] for k in proprio_keys) - target
    parts = []
    for k in proprio_keys:
        a = g[k]
        if k == "joint_pos" and deficit > 0:
            a = a[..., : a.shape[-1] - deficit]
        parts.append(a)
    proprio = torch.cat(parts, dim=-1)
    assert proprio.shape[-1] == target, (
        f"proprio dim {proprio.shape[-1]} != model proprio_dim {target}; proprio terms {proprio_keys} "
        f"(joint_pos trimmed by {max(deficit, 0)} for real-robot gripper dims)"
    )
    proprio = (proprio - bc["proprio_mean"]) / bc["proprio_std"]
    out = bc["model"](pc, proprio)
    mean = out[0] if isinstance(out, tuple) else out  # predict_std -> (mean, log_std)
    return mean * bc["action_std"] + bc["action_mean"]
