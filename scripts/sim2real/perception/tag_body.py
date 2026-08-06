"""A rigid body localized by AprilTags glued anywhere on it.

The peg's tag geometry is derived from CAD (``apriltag_peg_pose.FACES`` -> tag at the centre
of a cube face). That does not generalize: the peg-hole's USD origin is not its bbox centre,
and you glue tags wherever they fit. So a ``TagBody`` just stores the measured constant
``T_body_tag`` per tag id, with no assumption about where the tags sit.

Pose estimation reuses ``apriltag_peg_pose.estimate_peg_pose_ex`` unchanged -- that solver is
already body-agnostic (it takes a ``{tag_id: T_body_tag}`` dict and joint-PnPs every visible
corner against it), it just needed its tag size and dictionary parameterized.

JSON on disk (``calibrations/hole_tags.json``)::

    {"name": "peg_hole", "aruco_dict": "DICT_APRILTAG_16h5", "tag_size_m": 0.026,
     "dims_m": [0.06985, 0.06985, 0.04064],
     "tags": {"10": <4x4 row-major T_body_tag>, ...},
     "calib": {...provenance/residuals...}}
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import apriltag_peg_pose as apt

# ---- PegHole sim asset constants (Props/Custom/PegHole/metadata.yaml + peg_hole.obj) --------
# assembled_offset: the seated PEG's frame expressed in the HOLE frame. The peg's own
# assembled_offset is identity, so seated  =>  T_hole_peg = Trans(0, 0, ASSEMBLED_OFFSET_Z).
ASSEMBLED_OFFSET_Z = 0.014837
HOLE_DIMS_M = (0.06985, 0.06985, 0.04064)      # peg_hole.obj bbox (block, not the bore)
HOLE_BBOX_CENTER_Z = 0.00262                   # bbox centre in hole frame; origin is NOT centred
HOLE_TOP_Z = 0.02294                           # top face in hole frame (machined reference)
# sim success gate (metadata.yaml success_thresholds) -- calibration must land well inside this
HOLE_SUCCESS_POS_M = 0.0025
HOLE_SUCCESS_ORI_RAD = 0.025

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HOLE_TAGS = os.path.join(_HERE, "calibrations", "hole_tags.json")


def T_hole_peg_seated() -> np.ndarray:
    """Constant peg->hole transform when the peg is fully seated in the hole."""
    T = np.eye(4)
    T[2, 3] = ASSEMBLED_OFFSET_Z
    return T


def inv(T) -> np.ndarray:
    """Fast rigid 4x4 inverse."""
    T = np.asarray(T, dtype=np.float64)
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _stack(Ts):
    """(quats (N,4) hemisphere-aligned, positions (N,3)) from a list/array of 4x4 transforms.

    Batched on purpose: the fusion worker re-evaluates its running hole estimate inside the
    30 Hz publish loop, and per-sample scipy Rotation calls over a few hundred samples cost
    ~300 ms -- enough to stall the peg pipeline. Batched calls make it sub-millisecond.
    """
    A = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
    q = R.from_matrix(A[:, :3, :3]).as_quat()
    q[q @ q[0] < 0] *= -1                               # hemisphere-align before averaging
    return q, A[:, :3, 3]


def _weights(weights, n):
    w = np.ones(n) if weights is None else np.maximum(
        np.asarray(weights, dtype=np.float64).reshape(-1), 0.0)
    return np.ones(n) if not np.any(w > 0) else w


def _compose(q, t):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def chordal_mean(Ts, weights=None) -> np.ndarray:
    """Weighted quaternion(chordal) mean rotation + weighted mean position."""
    q, p = _stack(Ts)
    w = _weights(weights, len(q))
    qm = (w[:, None] * q).sum(0)
    return _compose(qm / np.linalg.norm(qm), (w[:, None] * p).sum(0) / w.sum())


def _residuals(q, p, M):
    """Per-sample (position m, rotation rad) deviation from the 4x4 ``M``."""
    qm = R.from_matrix(M[:3, :3]).as_quat()
    # geodesic angle between unit quats: 2*acos|<q, qm>|
    d_rot = 2.0 * np.arccos(np.clip(np.abs(q @ qm), 0.0, 1.0))
    return np.linalg.norm(p - M[:3, 3], axis=1), d_rot


def robust_mean(Ts, weights=None, iters=8, eps_m=2e-4, eps_deg=0.1) -> np.ndarray:
    """IRLS geodesic median: reweight by 1/residual so a flipped/mis-decoded sample drops out."""
    q, p = _stack(Ts)
    if len(q) == 1:
        return _compose(q[0], p[0])
    w0 = _weights(weights, len(q))
    w = w0
    eps_r = np.radians(eps_deg)
    M = None
    for _ in range(iters + 1):
        qm = (w[:, None] * q).sum(0)
        qm /= np.linalg.norm(qm)
        M = _compose(qm, (w[:, None] * p).sum(0) / w.sum())
        d_pos, d_rot = _residuals(q, p, M)
        # one shared weight per sample so position and rotation stay on the same rigid body
        w = w0 / np.maximum(np.maximum(d_pos / eps_m, d_rot / eps_r), 1.0)
    return M


def spread(Ts, ref=None):
    """(max position deviation in mm, max rotation deviation in deg) about ``ref``/their mean."""
    q, p = _stack(Ts)
    if len(q) < 2:
        return 0.0, 0.0
    M = chordal_mean(Ts) if ref is None else np.asarray(ref, dtype=np.float64)
    d_pos, d_rot = _residuals(q, p, M)
    return float(d_pos.max()) * 1000.0, float(np.degrees(d_rot.max()))


class TagBody:
    """A rigid body plus the measured constant pose of every tag stuck to it."""

    def __init__(self, name, tags, tag_size_m, aruco_dict="DICT_APRILTAG_16h5",
                 dims_m=HOLE_DIMS_M, calib=None):
        self.name = str(name)
        self.tags = {int(k): np.asarray(v, dtype=np.float64).reshape(4, 4)
                     for k, v in dict(tags).items()}
        self.tag_size_m = float(tag_size_m)
        self.aruco_dict = str(aruco_dict)
        self.dims_m = tuple(float(x) for x in dims_m)
        self.calib = dict(calib or {})

    # ---- persistence ----------------------------------------------------------------
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(name=d.get("name", "body"), tags=d["tags"], tag_size_m=d["tag_size_m"],
                   aruco_dict=d.get("aruco_dict", "DICT_APRILTAG_16h5"),
                   dims_m=d.get("dims_m", HOLE_DIMS_M), calib=d.get("calib"))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "name": self.name,
            "aruco_dict": self.aruco_dict,
            "tag_size_m": self.tag_size_m,
            "dims_m": list(self.dims_m),
            "tags": {str(i): self.tags[i].tolist() for i in sorted(self.tags)},
            "calib": self.calib,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return path

    # ---- detection ------------------------------------------------------------------
    @property
    def dict_id(self):
        return getattr(cv2.aruco, self.aruco_dict)

    def make_detector(self):
        """A detector for THIS body's tag family (ArucoDetector is not thread-safe -- one per
        consumer thread)."""
        return apt.make_detector(self.dict_id)

    def estimate(self, gray, detector, K, dist, prior=None, mode="joint"):
        """(T_cam_body | None, dbg, info) -- joint multi-tag PnP over this body's tags."""
        return apt.estimate_peg_pose_ex(gray, detector, K, dist, self.tags, prior=prior,
                                        mode=mode, tag_size=self.tag_size_m)

    def detect_tag_poses(self, gray, detector, K, dist, valid_ids=None):
        """[(tag_id, T_cam_tag)] for individual tags -- used during CALIBRATION, before the
        body geometry is known."""
        return apt._detect_tag_poses(gray, detector, K, dist,
                                     self.tags if valid_ids is None else valid_ids,
                                     tag_size=self.tag_size_m)

    def __repr__(self):
        return (f"TagBody({self.name}, {len(self.tags)} tags {sorted(self.tags)}, "
                f"{self.tag_size_m * 1000:.0f}mm, {self.aruco_dict})")
