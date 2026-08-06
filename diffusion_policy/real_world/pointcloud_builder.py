"""Build the PointNet observation cloud from a depth frame (see POINTCLOUD_EVAL.md).

depth (+ SAM2 label map) -> backproject -> base frame -> EE (wrist_3_link) frame ->
crop -> per-class budget sample -> (num_points, 4) = xyz + seg label.

Depth-source-agnostic: ``depth`` may come from FFS, RealSense hardware depth, or DA3.
Locked to ``pnocc_xl_residual_big_ee``: EE frame, 1024 pts, budget robot/peg/hole =
512/256/256, labels {robot:0.0, peg:-1.0, hole:+1.0}.

Conventions:
  * depth registered to the RGB used for SAM2 (same HxW, same intrinsics K).
  * K is the 3x3 color intrinsics; ``depth_scale`` converts raw depth -> metres.
  * ``T_cam_base`` is the 4x4 camera->base extrinsic (point_base = T_cam_base @ point_cam).
  * EE pose ``(ee_pos, ee_quat_wxyz)`` is wrist_3_link in base (from FK get_ee_pose).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# Default deployment constants for pnocc_xl_residual_big_ee.
SEG_LABELS = {"robot": 0.0, "peg": -1.0, "hole": 1.0}
DEFAULT_BUDGET = {0.0: 512, -1.0: 256, 1.0: 256}  # label -> target count (sums to 1024)
BG_LABEL = np.nan  # background / dropped points carry this in the label map


@dataclass
class CloudStats:
    """Diagnostics for one built cloud (printed by debug_pointcloud.py)."""

    n_raw_valid: int = 0          # valid depth pixels before labeling/crop
    n_after_crop: int = 0         # points surviving the workspace crop
    per_class_available: dict = field(default_factory=dict)  # label -> count found
    per_class_realized: dict = field(default_factory=dict)   # label -> count emitted
    short_classes: list = field(default_factory=list)        # classes that under-filled budget
    bbox_min: Optional[np.ndarray] = None   # EE-frame AABB of the emitted cloud
    bbox_max: Optional[np.ndarray] = None


def flying_pixel_mask(z: np.ndarray, thresh: float, ksize: int = 3) -> np.ndarray:
    """Boolean mask of flying pixels in a dense z/depth grid (True = drop).

    ToF flying pixels are mixed pixels on the ramp between a foreground edge and
    the background; the Orbbec's SW alignment (640x576 native depth -> 1280x960
    color grid) additionally interpolates across those discontinuities, smearing
    the ramp over 2-3 aligned pixels. In a point cloud they show up as streaks of
    3D points floating between the object and the background. A single
    adjacent-pixel diff can stay under threshold on every step of such a ramp, so
    we threshold the *windowed* range instead: max(z) - min(z) over a ``ksize`` x
    ``ksize`` neighborhood (morphological dilate/erode), computed only over valid
    (finite, >0) pixels so dropout regions neither trip the filter nor kill their
    valid neighbors.

    This flags the whole ramp plus ~1 px of legitimate edge on each side. A
    ``thresh`` of ~30 mm keeps steeply slanted real surfaces (an 80-degree-
    incidence surface at 0.7 m spans ~14 mm across a 3x3 window at Orbbec
    aligned resolution) while cutting fg/bg edge ramps, typically >100 mm.

    Args:
        z: (H, W) z-depth grid, any units; <=0 or non-finite = invalid.
        thresh: local-range threshold in the same units as ``z``.
    Returns:
        (H, W) bool; True where a valid pixel should be discarded. Invalid
        pixels are always False (they are already excluded downstream).
    """
    import cv2
    d = np.asarray(z, np.float32)
    valid = np.isfinite(d) & (d > 0)
    kernel = np.ones((ksize, ksize), np.uint8)
    # Sentinels exclude invalid pixels from the window max/min.
    hi = cv2.dilate(np.where(valid, d, np.float32(-1.0)), kernel)
    lo = cv2.erode(np.where(valid, d, np.float32(np.finfo(np.float32).max)), kernel)
    return valid & ((hi - lo) > thresh)


def backproject(depth: np.ndarray, K: np.ndarray, depth_scale: float = 1000.0):
    """Backproject a depth image to camera-frame points.

    Returns (points_cam (M,3) metres, pix_idx (M,) flat HxW index of each kept point)
    for pixels with finite, positive depth.
    """
    depth = np.asarray(depth, np.float64).squeeze()
    h, w = depth.shape
    z = depth / depth_scale
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    zf = z.reshape(-1)
    valid = np.isfinite(zf) & (zf > 0)
    idx = np.nonzero(valid)[0]
    return pts[idx], idx


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (M,3) points."""
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (T @ pts_h.T).T[:, :3]


def to_ee_frame(points_base: np.ndarray, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray:
    """Express base-frame points in the EE (wrist_3_link) frame.

    Mirrors sim ``quat_apply(quat_inv(ref_quat), p - ref_pos)``: p_ee = R_ee^T (p_base - ee_pos).
    """
    w, x, y, z = ee_quat_wxyz
    R_ee = R.from_quat([x, y, z, w]).as_matrix()  # scipy expects xyzw
    return (points_base - np.asarray(ee_pos)) @ R_ee  # (p - t) @ R == R^T (p - t)


def crop_aabb(points: np.ndarray, lo, hi, *extra):
    """Keep points inside the axis-aligned box [lo, hi]; filter ``extra`` arrays in lockstep."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    out = [points[keep]] + [a[keep] for a in extra]
    return tuple(out) if extra else out[0]


def budget_sample(points: np.ndarray, labels: np.ndarray, budget: dict,
                  pad: str = "repeat", rng: Optional[np.random.Generator] = None):
    """Sample a fixed per-class number of points.

    For each ``label -> target`` in ``budget``: draw ``target`` points of that class
    uniformly WITH replacement (i.i.d.), matching the sim training distribution. This
    holds whether or not the class has >= target points available.
    ``pad`` only matters when fewer than ``target`` are available:
      * "repeat" -- keep sampling with replacement to reach target (keeps N fixed; recommended)
      * "short"  -- emit only what's available, unrepeated (N may be < sum(budget))
    Returns (coords (N,3), labels (N,), per_available dict, short_list).
    """
    rng = rng or np.random.default_rng()
    out_pts, out_lab, available, short = [], [], {}, []
    for label, target in budget.items():
        sel = np.nonzero(labels == label)[0]
        available[label] = int(sel.size)
        if sel.size == 0:
            short.append(label)
            # Fully occluded class: emit its full budget as zero points (xyz=0, class label kept)
            # so the cloud stays a fixed size (sum(budget)) with the same per-class proportions,
            # instead of returning a smaller cloud the policy/accumulator can't ingest.
            out_pts.append(np.zeros((target, 3), np.float32))
            out_lab.append(np.full(target, label, np.float32))
            continue
        if pad == "short" and sel.size < target:
            pick = sel
            short.append(label)
        else:
            # Always sample WITH replacement to match the sim training distribution
            # (each of `target` points drawn i.i.d. from this class's available points).
            pick = rng.choice(sel, size=target, replace=True)
            if sel.size < target:
                short.append(label)
        out_pts.append(points[pick])
        out_lab.append(np.full(pick.shape[0], label, np.float32))
    coords = np.concatenate(out_pts, axis=0) if out_pts else np.zeros((0, 3), np.float32)
    labs = np.concatenate(out_lab, axis=0) if out_lab else np.zeros((0,), np.float32)
    return coords.astype(np.float32), labs.astype(np.float32), available, short


def build_cloud(depth, K, T_cam_base, ee_pos, ee_quat_wxyz, label_map=None,
                depth_scale: float = 1000.0, crop_lo=None, crop_hi=None,
                budget: Optional[dict] = None, pad: str = "repeat",
                rng: Optional[np.random.Generator] = None):
    """Full pipeline: depth (+ labels) -> (num_points, 4) EE-frame segmented cloud.

    Args:
        label_map: (H, W) per-pixel seg label aligned with ``depth`` (values in
            SEG_LABELS, background = NaN). If None, all kept points get label 0.0 and
            no per-class budgeting is applied (geometry-only debug mode).
        budget: label -> count. Defaults to DEFAULT_BUDGET when ``label_map`` is given.
        crop_lo/crop_hi: AABB bounds in the EE frame (metres) applied after transform.
    Returns:
        cloud (N, 4) float32 [x, y, z, label], CloudStats.
    """
    stats = CloudStats()
    pts_cam, pix_idx = backproject(depth, K, depth_scale)
    stats.n_raw_valid = int(pix_idx.size)

    pts_base = transform_points(pts_cam, np.asarray(T_cam_base, np.float64))
    pts_ee = to_ee_frame(pts_base, ee_pos, ee_quat_wxyz)

    if label_map is not None:
        labels = np.asarray(label_map, np.float32).reshape(-1)[pix_idx]
        keep = np.isfinite(labels)  # drop background pixels
        pts_ee, labels = pts_ee[keep], labels[keep]
    else:
        labels = np.zeros(pts_ee.shape[0], np.float32)

    if crop_lo is not None and crop_hi is not None:
        pts_ee, labels = crop_aabb(pts_ee, crop_lo, crop_hi, labels)
    stats.n_after_crop = int(pts_ee.shape[0])

    if label_map is None:
        coords, labs = pts_ee.astype(np.float32), labels
        stats.per_class_available = {0.0: int(coords.shape[0])}
        stats.per_class_realized = {0.0: int(coords.shape[0])}
    else:
        budget = budget or DEFAULT_BUDGET
        coords, labs, available, short = budget_sample(pts_ee, labels, budget, pad, rng)
        stats.per_class_available = available
        stats.per_class_realized = {lab: int((labs == lab).sum()) for lab in budget}
        stats.short_classes = short

    if coords.shape[0]:
        stats.bbox_min, stats.bbox_max = coords.min(0), coords.max(0)
    cloud = np.concatenate([coords, labs[:, None]], axis=1).astype(np.float32)
    return cloud, stats


# --- torch / GPU pipeline -------------------------------------------------------------------
# Mirrors the numpy path above op-for-op (same frames, conventions, budget) but keeps everything
# on the device so SAM2 masks + depth never round-trip to host. depth/label_map are torch tensors
# already on the device; K / T_cam_base / ee_* stay small numpy/array inputs (read as scalars).


def backproject_torch(depth, K, depth_scale: float = 1000.0):
    """torch twin of ``backproject``: (H,W) depth tensor -> (pts_cam (M,3), pix_idx (M,))."""
    import torch
    depth = depth.squeeze()
    h, w = depth.shape
    z = depth.to(torch.float32) / depth_scale
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32), indexing="xy")
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    zf = z.reshape(-1)
    valid = torch.isfinite(zf) & (zf > 0)
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    return pts[idx], idx


def transform_points_torch(points, T):
    """Apply a 4x4 homogeneous transform to (M,3) torch points: p @ R^T + t."""
    import torch
    T = torch.as_tensor(T, dtype=points.dtype, device=points.device)
    return points @ T[:3, :3].T + T[:3, 3]


def to_ee_frame_torch(points_base, ee_pos, ee_quat_wxyz):
    """torch twin of ``to_ee_frame``: p_ee = R_ee^T (p_base - ee_pos), quat is wxyz."""
    import torch
    w, x, y, z = (float(v) for v in ee_quat_wxyz)
    R_ee = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=points_base.dtype, device=points_base.device)
    t = torch.as_tensor(ee_pos, dtype=points_base.dtype, device=points_base.device)
    return (points_base - t) @ R_ee  # (p - t) @ R == R^T (p - t)


def crop_aabb_torch(points, lo, hi, *extra):
    """Keep points inside [lo, hi]; filter ``extra`` tensors in lockstep."""
    import torch
    lo = torch.as_tensor(lo, dtype=points.dtype, device=points.device)
    hi = torch.as_tensor(hi, dtype=points.dtype, device=points.device)
    keep = ((points >= lo) & (points <= hi)).all(dim=1)
    out = [points[keep]] + [a[keep] for a in extra]
    return tuple(out) if extra else out[0]


def budget_sample_torch(points, labels, budget: dict, pad: str = "repeat", generator=None):
    """torch twin of ``budget_sample``: fixed per-class sampling on-device."""
    import torch
    dev = points.device
    out_pts, out_lab, available, short = [], [], {}, []
    for label, target in budget.items():
        sel = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        n = int(sel.numel())
        available[label] = n
        if n == 0:
            short.append(label)
            # Fully occluded class: emit its full budget as zero points (xyz=0, class label kept)
            # so the cloud stays a fixed size (sum(budget)) with the same per-class proportions,
            # instead of returning a smaller cloud the policy/accumulator can't ingest.
            out_pts.append(torch.zeros((target, 3), dtype=torch.float32, device=dev))
            out_lab.append(torch.full((target,), float(label), dtype=torch.float32, device=dev))
            continue
        if pad == "short" and n < target:
            pick = sel
            short.append(label)
        else:
            # Always sample WITH replacement to match the sim training distribution
            # (each of `target` points drawn i.i.d. from this class's available points).
            pick = sel[torch.randint(0, n, (target,), device=dev, generator=generator)]
            if n < target:
                short.append(label)
        out_pts.append(points[pick])
        out_lab.append(torch.full((pick.numel(),), float(label), dtype=torch.float32, device=dev))
    coords = torch.cat(out_pts, 0) if out_pts else torch.zeros((0, 3), dtype=torch.float32, device=dev)
    labs = torch.cat(out_lab, 0) if out_lab else torch.zeros((0,), dtype=torch.float32, device=dev)
    return coords.to(torch.float32), labs, available, short


def assemble_cloud_torch(pts_cam, labels, T_cam_base, ee_pos, ee_quat_wxyz, label_mode=True,
                         crop_lo=None, crop_hi=None, budget: Optional[dict] = None,
                         pad: str = "repeat", generator=None):
    """Shared tail of the on-device cloud build: camera-frame points (+ per-point labels) ->
    base -> EE frame -> crop -> per-class budget -> (cloud (N,4), CloudStats).

    Factored out so both the depth-backprojection path (``build_cloud_torch``) and a sensor that
    emits points directly (e.g. Orbbec's ``PointCloudFilter``) feed the SAME transform/crop/budget
    logic. ``pts_cam`` is (M,3) in the camera optical frame; ``labels`` is (M,) with SEG_LABELS
    values and ``NaN`` for background. ``label_mode=False`` keeps every point with label 0.0 and
    skips budgeting (geometry-only debug)."""
    import torch
    stats = CloudStats()
    stats.n_raw_valid = int(pts_cam.shape[0])

    pts_base = transform_points_torch(pts_cam, T_cam_base)
    pts_ee = to_ee_frame_torch(pts_base, ee_pos, ee_quat_wxyz)

    if label_mode:
        keep = torch.isfinite(labels)  # drop background points
        pts_ee, labels = pts_ee[keep], labels[keep]

    if crop_lo is not None and crop_hi is not None:
        pts_ee, labels = crop_aabb_torch(pts_ee, crop_lo, crop_hi, labels)
    stats.n_after_crop = int(pts_ee.shape[0])

    if not label_mode:
        coords, labs = pts_ee.to(torch.float32), labels
        stats.per_class_available = {0.0: int(coords.shape[0])}
        stats.per_class_realized = {0.0: int(coords.shape[0])}
    else:
        budget = budget or DEFAULT_BUDGET
        coords, labs, available, short = budget_sample_torch(pts_ee, labels, budget, pad, generator)
        stats.per_class_available = available
        stats.per_class_realized = {lab: int((labs == lab).sum()) for lab in budget}
        stats.short_classes = short

    if coords.shape[0]:
        stats.bbox_min = coords.min(0).values.detach().cpu().numpy()
        stats.bbox_max = coords.max(0).values.detach().cpu().numpy()
    cloud = torch.cat([coords, labs[:, None]], dim=1).to(torch.float32)
    return cloud, stats


def build_cloud_torch(depth, K, T_cam_base, ee_pos, ee_quat_wxyz, label_map=None,
                      depth_scale: float = 1000.0, crop_lo=None, crop_hi=None,
                      budget: Optional[dict] = None, pad: str = "repeat", generator=None):
    """On-device twin of ``build_cloud``: depth tensor (+ label_map tensor) -> (cloud (N,4)
    float32 tensor, CloudStats). bbox stats are pulled to host (small); the cloud stays on-device."""
    import torch
    pts_cam, pix_idx = backproject_torch(depth, K, depth_scale)
    if label_map is not None:
        labels = label_map.reshape(-1)[pix_idx]
    else:
        labels = torch.zeros(pts_cam.shape[0], dtype=torch.float32, device=pts_cam.device)
    return assemble_cloud_torch(
        pts_cam, labels, T_cam_base, ee_pos, ee_quat_wxyz, label_mode=(label_map is not None),
        crop_lo=crop_lo, crop_hi=crop_hi, budget=budget, pad=pad, generator=generator)
