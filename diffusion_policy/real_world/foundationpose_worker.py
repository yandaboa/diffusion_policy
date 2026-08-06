"""FoundationPose 6-DoF object-pose worker (runs in the ``foundationpose`` conda env).

Launched as a subprocess by ``foundationpose_client.py`` (mirrors ``ffs_worker.py``). Reads a
color frame + color-aligned metric depth + intrinsics from shared memory, runs FoundationPose
``register`` (first frame, needs an object mask) or ``track_one`` (subsequent frames), and
writes the 4x4 object-in-camera pose back. State-byte protocol must match the client.

The heavy FoundationPose deps (torch 2.0+cu118, pytorch3d, nvdiffrast, mycpp) live only in the
``foundationpose`` env, so this worker runs under ``FOUNDATIONPOSE_PYTHON`` and the calling
process needs none of them.

GOTCHA (see setup notes): the FoundationPose repo ROOT must be on sys.path, NOT
``<repo>/mycpp/build`` -- ``Utils.py`` does ``import mycpp.build.mycpp`` (namespace package),
and adding mycpp/build shadows it, silently making ``mycpp = None``.

Usage (normally invoked by the client, not by hand):
  python foundationpose_worker.py --prefix fp_xxxx --fp-repo <repo> --mesh <obj> --max-h .. --max-w ..
  python foundationpose_worker.py --prefix fp_xxxx --mock         # skip FoundationPose (handshake test)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# State-byte protocol (must match foundationpose_client.py)
_IDLE, _FRAME_READY, _POSE_READY, _WORKER_READY, _STOP = (
    np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(10), np.uint8(255))

# meta layout (uint32): [h, w, mode(0=track,1=register), _, _, _, _, _]
_MODE_TRACK, _MODE_REGISTER = 0, 1


def _build_estimator(fp_repo: str, mesh_path: str, est_iter: int, track_iter: int, debug_dir: str):
    """Load the mesh + FoundationPose networks. Returns (est, K-agnostic estimator)."""
    # FP repo ROOT only (see GOTCHA above); Utils.py appends its own code_dir.
    sys.path.insert(0, fp_repo)
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    mesh = trimesh.load(mesh_path, process=False)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx, debug=0, debug_dir=debug_dir)
    return est


def _mock_pose(depth: np.ndarray, mask: "np.ndarray | None", K: np.ndarray) -> np.ndarray:
    """Plumbing-test pose: identity rotation, translation = backprojected mask/image centroid."""
    pose = np.eye(4, dtype=np.float64)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        u, v = xs.mean(), ys.mean()
        z = np.median(depth[mask & (depth > 0)]) if (depth[mask] > 0).any() else 0.5
    else:
        v, u = np.array(depth.shape) / 2.0
        valid = depth > 0
        z = np.median(depth[valid]) if valid.any() else 0.5
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose[:3, 3] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--fp-repo", default=os.environ.get("FOUNDATIONPOSE_REPO", ""))
    ap.add_argument("--mesh", default="")
    ap.add_argument("--est-iter", type=int, default=5)
    ap.add_argument("--track-iter", type=int, default=2)
    ap.add_argument("--max-h", type=int, default=1200)
    ap.add_argument("--max-w", type=int, default=1920)
    ap.add_argument("--debug-dir", default="/tmp/fp_worker_dbg")
    ap.add_argument("--mock", action="store_true", help="skip FoundationPose (handshake test)")
    args = ap.parse_args()
    p = args.prefix
    MH, MW = args.max_h, args.max_w

    shms = {name: SharedMemory(name=f"{p}_{name}") for name in
            ("state", "meta", "params", "rgb", "depth", "mask", "pose")}
    state = np.ndarray(1, np.uint8, buffer=shms["state"].buf)
    meta = np.ndarray(8, np.uint32, buffer=shms["meta"].buf)
    params = np.ndarray(4, np.float64, buffer=shms["params"].buf)   # fx, fy, cx, cy
    rgb_buf = np.ndarray(MH * MW * 3, np.uint8, buffer=shms["rgb"].buf)
    depth_buf = np.ndarray(MH * MW, np.float32, buffer=shms["depth"].buf)
    mask_buf = np.ndarray(MH * MW, np.uint8, buffer=shms["mask"].buf)
    pose_buf = np.ndarray(16, np.float64, buffer=shms["pose"].buf)

    est = None
    if not args.mock:
        os.makedirs(args.debug_dir, exist_ok=True)
        est = _build_estimator(args.fp_repo, args.mesh, args.est_iter, args.track_iter, args.debug_dir)
    state[0] = _WORKER_READY
    print(f"[fp_worker] ready (mock={args.mock}, mesh={os.path.basename(args.mesh)})", flush=True)

    try:
        while True:
            if int(state[0]) == int(_STOP):
                break
            if int(state[0]) != int(_FRAME_READY):
                time.sleep(0.001)
                continue
            h, w, mode = int(meta[0]), int(meta[1]), int(meta[2])
            fx, fy, cx, cy = params
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            rgb = rgb_buf[:h * w * 3].reshape(h, w, 3).copy()
            depth = depth_buf[:h * w].reshape(h, w).copy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            mask = mask_buf[:h * w].reshape(h, w).astype(bool).copy() if mode == _MODE_REGISTER else None

            if args.mock:
                pose = _mock_pose(depth, mask, K)
            elif mode == _MODE_REGISTER:
                pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                    iteration=args.est_iter)
            else:
                pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_iter)

            pose_buf[:] = np.asarray(pose, np.float64).reshape(-1)
            state[0] = _POSE_READY
    finally:
        for s in shms.values():
            s.close()


if __name__ == "__main__":
    main()
