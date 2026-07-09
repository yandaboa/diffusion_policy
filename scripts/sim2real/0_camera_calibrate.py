import os
import sys
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from perception.multi_camera_wrapper import MultiCameraWrapper
from perception.pcd_utils import *


# ============================================================================
# ChArUco-board calibration
# ----------------------------------------------------------------------------
# The world frame is rigidly anchored to the ChArUco board. OpenCV's native
# board frame already places its origin (0,0,0) at the printed TOP-LEFT corner
# (the corner adjacent to marker ids 0 and 5), with +X along the top edge, +Y
# down the left edge and +Z out of the board plane. That is exactly the world
# root we want ("upper-left corner near id 5"), so world == board frame and no
# re-origin transform is applied.
#
# solvePnP gives T_cam_board (board->camera: p_cam = R @ p_board + t). The saved
# extrinsic is its inverse, T_world_cam (camera->world: p_world = T @ p_cam), so
# camera_base_pos is the camera's position in the board/world frame -- matching
# the convention used by the single-marker ArUco path below.
# ============================================================================
def _charuco_board(args):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        float(args.square_length), float(args.marker_length), dictionary)
    return board


def _charuco_pose_once(camera, board):
    """Detect the board in one frame and solvePnP. Returns a dict or None."""
    frames = camera.read_camera()
    frame = frames["rgb"]  # HxWx3 RGB
    K = np.asarray(camera.calibration["intrinsics"]["rgb"]["cameraMatrix"], float)
    dist = np.asarray(camera.calibration["intrinsics"]["rgb"]["distCoeffs"], float).ravel()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 8:
        return None
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    if obj is None or len(obj) < 8:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec[:, 0]
    return dict(T_cam_board=T_cam_board, rvec=rvec, tvec=tvec,
                ch_corners=ch_corners, ch_ids=ch_ids, m_corners=m_corners,
                m_ids=m_ids, frame=frame, K=K, dist=dist, depth=frames.get("depth"))


def _measure_square_from_depth(res, board, square_length):
    """Cross-check the assumed square_length against the camera's own depth.

    Backproject each detected chessboard corner with the aligned depth, then
    compare camera-frame distances to the (planar) board-frame distances. The
    median ratio scales square_length -> a measured square size, independent of
    any ruler. Returns metres or None if depth is unavailable/too sparse.
    """
    depth = res["depth"]
    if depth is None:
        return None
    K = res["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    all_obj = board.getChessboardCorners()
    ids = res["ch_ids"].flatten()
    px = res["ch_corners"].reshape(-1, 2)
    H, W = depth.shape
    P, O = [], []
    for k, cid in enumerate(ids):
        u, v = float(px[k][0]), float(px[k][1])
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        z = float(depth[vi, ui]) / 1000.0
        if z <= 0.1 or z > 5.0:
            continue
        P.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        O.append(all_obj[cid])
    if len(P) < 8:
        return None
    P, O = np.asarray(P), np.asarray(O)
    ratios = []
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            do = np.linalg.norm(O[i] - O[j])
            if do < 1e-3:
                continue
            ratios.append(np.linalg.norm(P[i] - P[j]) / do)
    if not ratios:
        return None
    return float(square_length * np.median(ratios))


def calibrate_charuco(multi_camera_wrapper, args):
    board = _charuco_board(args)
    cameras = multi_camera_wrapper._all_cameras
    if not cameras:
        raise RuntimeError("No cameras found for ChArUco calibration.")

    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "perception/calibrations/")
    os.makedirs(calib_path, exist_ok=True)
    stamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    final_calib_dict = []
    for camera in cameras:
        serial = camera._serial_number
        print(f"\n=== ChArUco calibration: camera {serial} ===")
        Ts, res0, meas = [], None, []
        for r in range(args.rounds):
            res = _charuco_pose_once(camera, board)
            if res is None:
                print(f"  round {r}: board not detected (need >=8 corners), skipping")
                continue
            T_world_cam = np.linalg.inv(res["T_cam_board"])  # camera->world
            Ts.append(T_world_cam)
            if res0 is None:
                res0 = res
            m = _measure_square_from_depth(res, board, args.square_length)
            if m is not None:
                meas.append(m)
            print(f"  round {r}: {len(res['ch_ids'])} corners | "
                  f"cam pos in world = {T_world_cam[:3, 3].round(4)}")
        if not Ts:
            print(f"  !! board never detected for {serial}; skipping camera")
            continue

        # average translation; average + re-orthonormalize rotation
        pos = np.mean([T[:3, 3] for T in Ts], axis=0)
        Rbar = np.mean([T[:3, :3] for T in Ts], axis=0)
        U, _, Vt = np.linalg.svd(Rbar)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = Rm
        T_world_cam[:3, 3] = pos

        K = res0["K"]
        intr = camera.calibration["intrinsics"]["rgb"]
        rgb = res0["frame"]
        final_calib_dict.append({
            "camera_serial_number": serial,
            "intrinsics_raw": K.tolist(),
            "extrinsics_raw": T_world_cam.tolist(),
            "intrinsics": {
                "fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2],
                "height": rgb.shape[0], "width": rgb.shape[1],
                "fovy": camera._fovy,
                "coeffs": np.asarray(intr["distCoeffs"]).ravel().tolist(),
            },
            "camera_base_ori": Rm.tolist(),
            "camera_base_pos": pos.reshape(3, 1).tolist(),
            "world_frame": "charuco_board_top_left_corner(id5)",
            "charuco": {"dict": args.dict, "squares_x": args.squares_x,
                        "squares_y": args.squares_y,
                        "square_length": args.square_length,
                        "marker_length": args.marker_length},
        })

        # report + annotate
        dist_cam = float(np.linalg.norm(pos))
        print(f"  --> camera position in world (board top-left) = {pos.round(4)} m")
        print(f"  --> distance from world origin              = {dist_cam:.4f} m")
        if meas:
            mm = float(np.median(meas))
            print(f"  --> depth-measured square length ~= {mm*1000:.1f} mm "
                  f"(assumed {args.square_length*1000:.1f} mm). "
                  f"Translation scales linearly with square_length.")

        ann = cv2.cvtColor(res0["frame"], cv2.COLOR_RGB2BGR)
        cv2.aruco.drawDetectedMarkers(ann, res0["m_corners"], res0["m_ids"])
        cv2.drawFrameAxes(ann, K, res0["dist"], res0["rvec"], res0["tvec"],
                          args.square_length * 3, 4)
        ann_path = os.path.join(calib_path, f"{stamp}_charuco_{serial}.png")
        cv2.imwrite(ann_path, ann)
        print(f"  --> annotated axes (origin at world root) -> {ann_path}")

    if not final_calib_dict:
        raise RuntimeError("ChArUco calibration produced no results.")

    out_main = os.path.join(calib_path, f"{stamp}_charuco.json")
    json.dump(final_calib_dict, open(out_main, "w"), indent=2)
    json.dump(final_calib_dict,
              open(os.path.join(calib_path, "most_recent_charuco_calib.json"), "w"),
              indent=2)
    print(f"\nSaved ChArUco calibration -> {out_main}")
    print(f"               (and        -> {os.path.join(calib_path, 'most_recent_charuco_calib.json')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["realsense", "orbbec"], default="realsense",
                        help="Camera backend to use.")
    parser.add_argument("--board", choices=["charuco", "aruco"], default="charuco",
                        help="charuco: ChArUco-board world frame (default). "
                             "aruco: legacy single-marker pose.")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Calibration rounds to average over.")
    # ChArUco board geometry (defaults match the 11x11 / 24in board).
    parser.add_argument("--dict", default="DICT_4X4_100",
                        help="ArUco predefined dictionary name.")
    parser.add_argument("--squares_x", type=int, default=11)
    parser.add_argument("--squares_y", type=int, default=11)
    parser.add_argument("--square_length", type=float, default=0.0554,
                        help="Chessboard square side in metres (sets translation scale). "
                             "Default 24in/11; pass the measured value for accurate scale.")
    parser.add_argument("--marker_length", type=float, default=0.04,
                        help="ArUco marker side in metres (detection only).")
    args = parser.parse_args()

    # Number of calibration rounds
    NUM_CALIBRATION_ROUNDS = args.rounds

    # gather cameras
    multi_camera_wrapper = MultiCameraWrapper(rgb=True, depth=True, ir=False, high_res_rgb=False, align="rgb", type=args.camera)
    num_cameras = multi_camera_wrapper.num_cameras
    print(f"Number of cameras: {num_cameras}")

    if args.board == "charuco":
        calibrate_charuco(multi_camera_wrapper, args)
        try:
            multi_camera_wrapper.disable_cameras()
        except Exception:
            pass
        sys.exit(0)

    # calibrate using aruco tag
    pcds = []
    all_calib_dicts = []
    
    for round_idx in range(NUM_CALIBRATION_ROUNDS):
        print(f"\nPerforming calibration round {round_idx + 1}/{NUM_CALIBRATION_ROUNDS}")
        round_calib_dict = []
        
        for camera in multi_camera_wrapper._all_cameras:
            intrinsics = camera.calibration["intrinsics"]["rgb"]["cameraMatrix"]
            print(f"Intrinsics:\n{intrinsics}")

            marker_size = 0.15
            tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
                camera, marker_size=marker_size, verbose=True
            )
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotmat
            extrinsics[:3, 3:] = tvec
            print(f"Extrinsics:\n{extrinsics}")
            # camera frame -> aruco frame
            extrinsics_inv = np.linalg.inv(extrinsics)

            frames = camera.read_camera()
            rgb = frames["rgb"]
            depth = frames["depth"]

            # TODO: insert aruco offset to base
            aruco_offset = np.array(
                [
                    0.24,
                    0.0,
                    0.0,
                ]
            )

            extrinsics_inv[:3, 3] += aruco_offset

            round_calib_dict.append(
                {
                    "camera_serial_number": camera._serial_number,
                    "intrinsics_raw": intrinsics.tolist(),
                    "extrinsics_raw": extrinsics_inv.tolist(),
                    "intrinsics": {
                        "fx": intrinsics[0, 0],
                        "fy": intrinsics[1, 1],
                        "ppx": intrinsics[0, 2],
                        "ppy": intrinsics[1, 2],
                        "height": rgb.shape[0],
                        "width": rgb.shape[1],
                        "fovy": camera._fovy,
                        "coeffs": camera.calibration["intrinsics"]["rgb"][
                            "distCoeffs"
                        ].tolist(),
                    },
                    "camera_base_ori": extrinsics_inv[:3, :3].tolist(),
                    "camera_base_pos": extrinsics_inv[:3, 3:].tolist(),
                }
            )

            if round_idx == 0:  # Only collect point cloud data in first round
                points = depth_to_points(depth, intrinsics, extrinsics_inv, depth_scale=1000.0)
                colors = rgb.reshape(-1, 3) / 255.0
                points, colors = crop_points(points, colors=colors, crop_min=-2*np.ones(3), crop_max=2*np.ones(3))
                pcds.append(points_to_pcd(points, colors=colors))
        
        all_calib_dicts.append(round_calib_dict)

    # Average the calibration results
    final_calib_dict = []
    for camera_idx in range(num_cameras):
        # Collect all measurements for this camera
        camera_measurements = [round_dict[camera_idx] for round_dict in all_calib_dicts]
        
        # Average the extrinsics
        avg_extrinsics_raw = np.mean([np.array(m["extrinsics_raw"]) for m in camera_measurements], axis=0)
        avg_camera_base_ori = np.mean([np.array(m["camera_base_ori"]) for m in camera_measurements], axis=0)
        avg_camera_base_pos = np.mean([np.array(m["camera_base_pos"]) for m in camera_measurements], axis=0)
        
        # Use the first measurement for intrinsics (these shouldn't change)
        first_measurement = camera_measurements[0]
        
        final_calib_dict.append({
            "camera_serial_number": first_measurement["camera_serial_number"],
            "intrinsics_raw": first_measurement["intrinsics_raw"],
            "extrinsics_raw": avg_extrinsics_raw.tolist(),
            "intrinsics": first_measurement["intrinsics"],
            "camera_base_ori": avg_camera_base_ori.tolist(),
            "camera_base_pos": avg_camera_base_pos.tolist(),
        })

    current_time_date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception/calibrations/")
    os.makedirs(calib_path, exist_ok=True)
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"{current_time_date}.json"), "w"),
    )
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"most_recent_calib.json"), "w"),
    )
    print(f"Saved calibration at {os.path.join(calib_path,f'perception/logs/aruco/most_recent_calib.json')}")

    x = np.zeros((1, 3))
    for d in np.arange(0, 1, 0.1):
        x[:, 0] = d
        pcds.append(points_to_pcd(x, colors=[[255.0, 0.0, 0.0]]))
        y = np.zeros((1, 3))
        y[:, 1] = d
        pcds.append(points_to_pcd(y, colors=[[0.0, 255.0, 0.0]]))
        z = np.zeros((1, 3))
        z[:, 2] = d
        pcds.append(points_to_pcd(z, colors=[[0.0, 0.0, 255.0]]))

    visualize_pcds(pcds)
