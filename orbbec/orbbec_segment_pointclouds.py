"""
Orbbec (Femto Bolt) + SAM2 foreground point-cloud extraction.

The Orbbec sibling of `segment_pointclouds.py`. Instead of running Fast-
FoundationStereo on a D455 IR pair, this streams RGB-D straight from the Femto
Bolt and uses the point cloud the *camera* already gives us. SAM2 segments the
object(s) of interest on the color frame, and that mask keeps ONLY the matching
points -- table / floor / curtains / background fall away.

Runs in the `foundstereo` conda env (has sam2 + open3d + the FFS Utils helpers).
The Orbbec Python SDK is the `pyorbbecsdk2` wheel (imports as `pyorbbecsdk`).

Why this works cleanly on the Femto Bolt:
  * The Bolt has no hardware depth-to-color alignment, so we align in software
    with `AlignFilter(COLOR_STREAM)` -- depth is resampled onto the 1280x720
    color grid, so every (u,v) color pixel has a depth at depth[v,u].
  * `PointCloudFilter` then emits a DENSE (H*W, 6) array (x,y,z,r,g,b), row-major
    over that same color grid. So a SAM2 mask in color-image space indexes the
    Orbbec point cloud directly -- reshape to (H,W,6), keep mask & valid-z rows.
  * Bolt positions come out in millimeters; we convert to meters on the way out
    so the clouds match the D455 ones (meters, camera frame).

Unlike the offline `segment_pointclouds.py` (SAM2 *video* predictor over a
recorded directory), this uses the SAM2 *image* predictor on the live frame:
prompt once, segment that frame. Re-prompt per run since the scene differs.

Usage:
  # interactive: click the object(s) on the popped-up color frame, close window
  (foundstereo)$ python orbbec_segment_pointclouds.py --out_dir tmp/orbbec_seg
  # headless / scripted: give the click(s) directly
  (foundstereo)$ python orbbec_segment_pointclouds.py --out_dir tmp/orbbec_seg --point 640,360
  # extra clicks: positive ';'-separated, negatives via --neg; live 3D view via --vis
  (foundstereo)$ python orbbec_segment_pointclouds.py --out_dir <d> --point 640,360;700,400 --neg 100,100 --vis 1

Outputs (into <out_dir>/):
  cloud_fg_NNNNN.ply   foreground-only point cloud (color frame, meters)
  color_NNNNN.png      raw color frame (QA)
  mask_NNNNN.png       binary mask (QA)
  overlay_NNNNN.png    mask drawn on the color frame (QA)
  segment_meta.json    intrinsics, prompts, depth range
"""
import os, sys, json, time, argparse

REPO = os.path.dirname(os.path.realpath(__file__))
FFS = os.path.join(REPO, 'Fast-FoundationStereo')
sys.path.append(FFS)

import numpy as np
import cv2
import torch

from Utils import toOpen3dCloud, o3d
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Core Orbbec Femto Bolt camera API (open / warmup / capture / intrinsics / cloud).
from orbbec_camera import (
    open_camera, warmup_autoexposure, capture_aligned,
    color_intrinsics, make_pointcloud_filter, orbbec_pointcloud,
)


# ----------------------------------------------------------------------------
# prompt parsing / interactive click picking (mirrors segment_pointclouds.py)
# ----------------------------------------------------------------------------
def parse_points(s):
    """'640,360;700,400' -> [[640,360],[700,400]]; '' -> []"""
    if not s:
        return []
    out = []
    for pair in s.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(',')
        out.append([float(x), float(y)])
    return out


def pick_points_interactive(img_rgb):
    """Show the color frame; LEFT-click = object (positive), RIGHT-click =
    background (negative). Close the window when done. Returns (pos, neg)."""
    import matplotlib.pyplot as plt
    print('[orbbec-seg] LEFT-click the object(s) of interest (green), '
          'RIGHT-click background (red), then close the window...')
    pos, neg = [], []
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(img_rgb)
    ax.set_title('LEFT-click = object (green)  |  RIGHT-click = background (red)  |  '
                 'close window when done')

    def onclick(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:        # left -> positive
            pos.append([float(event.xdata), float(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'o', color='lime', ms=8, mec='black')
        elif event.button == 3:      # right -> negative
            neg.append([float(event.xdata), float(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'x', color='red', ms=10, mew=3)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()                       # blocks until the window is closed
    fig.canvas.mpl_disconnect(cid)
    if not pos:
        raise RuntimeError('no positive (object) point clicked; pass --point x,y instead')
    return pos, neg


# ----------------------------------------------------------------------------
# robot joint state (so sim can reset to the same pose as this capture)
# ----------------------------------------------------------------------------
def read_robot_state(robot_ip):
    """Read the UR5e's current 6-DOF joint angles (radians) + TCP pose over RTDE.
    Best-effort: returns None (with a warning) if ur_rtde is missing or the robot
    is unreachable, so the perception path still works without the arm."""
    try:
        from rtde_receive import RTDEReceiveInterface
    except Exception as e:
        print(f'[orbbec-seg] ur_rtde not importable ({e}); skipping robot state')
        return None
    try:
        rtde_r = RTDEReceiveInterface(robot_ip)
        q = np.array(rtde_r.getActualQ(), dtype=float)        # 6 joints, radians
        tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)  # x,y,z,rx,ry,rz
        rtde_r.disconnect()
    except Exception as e:
        print(f'[orbbec-seg] could not read robot at {robot_ip} ({e}); '
              f'skipping robot state')
        return None
    print(f'[orbbec-seg] robot joints (rad): {np.round(q, 4).tolist()}')
    return {
        'robot_ip': robot_ip,
        'joints_rad': q.tolist(),
        'joints_deg': np.rad2deg(q).tolist(),
        'tcp_pose': tcp.tolist(),     # meters + axis-angle (rad), UR base frame
        'timestamp': time.time(),
    }


# ----------------------------------------------------------------------------
# fanning / flying-pixel reduction
# ----------------------------------------------------------------------------
def edge_keep_image(z_mm, thresh_mm, ksize):
    """Bool image, False on depth discontinuities. A ToF pixel straddling the
    object silhouette and the table behind it gets an averaged depth and back-
    projects to a 'flying pixel' in the fan. We flag any pixel whose local
    depth-valid neighborhood spans more than thresh_mm (i.e. sits on a depth
    step) and drop it. Considers ALL valid depth (not just the object mask) so
    the foreground/background step at the silhouette is seen."""
    kernel = np.ones((ksize, ksize), np.uint8)
    z = z_mm.astype(np.float32)
    valid = z > 0
    local_max = cv2.dilate(np.where(valid, z, 0.0), kernel)
    big = float(z.max()) + 1.0 if z.size else 1.0
    local_min = cv2.erode(np.where(valid, z, big), kernel)   # min over valid only
    spread = local_max - local_min
    return spread <= thresh_mm


def radius_outlier_clean(pcd, nb_points, radius_m):
    """Drop points with fewer than nb_points neighbors within radius_m -- the
    residual sparse flyers the edge filter misses."""
    if len(pcd.points) == 0 or nb_points <= 0:
        return pcd
    pcd, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius_m)
    return pcd


# ----------------------------------------------------------------------------
# SAM2 (image predictor, single live frame)
# ----------------------------------------------------------------------------
def load_sam2(checkpoint, model_cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model), device


def segment(predictor, color_rgb, pos, neg, device):
    """Prompt SAM2 with positive/negative clicks; return the best (H,W) bool mask."""
    points = np.array(pos + neg, np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == 'cuda' \
        else torch.no_grad()
    with torch.inference_mode(), autocast:
        predictor.set_image(color_rgb)                   # SAM2 expects RGB
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(bool)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--serial', default=None, help='Orbbec serial (default: first device)')
    ap.add_argument('--checkpoint',
                    default=f'{REPO}/weights/sam2/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--model_cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    ap.add_argument('--point', default='', help='positive click(s) "x,y" or "x1,y1;x2,y2"')
    ap.add_argument('--neg', default='', help='negative click(s), same format')
    ap.add_argument('--color_w', type=int, default=1280)
    ap.add_argument('--color_h', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--exposure', type=int, default=None,
                    help='fix color exposure (disables AE); default None = auto-exposure')
    ap.add_argument('--warmup_seconds', type=float, default=3.0,
                    help='max time to let color auto-exposure settle before capturing')
    ap.add_argument('--robot_ip', default='192.168.1.10',
                    help='UR5e IP to read joint state from (for sim reset)')
    ap.add_argument('--read_robot', type=int, default=1,
                    help='read + save the robot joint state (0 to skip)')
    ap.add_argument('--zmin', type=float, default=0.1, help='min depth to keep (m)')
    ap.add_argument('--zmax', type=float, default=2.0, help='max depth to keep (m)')
    # --- fanning / flying-pixel reduction (all tunable; 0 disables a stage) ---
    ap.add_argument('--erode_mask', type=int, default=3,
                    help='erode SAM2 mask by N px to drop the mixed-pixel silhouette ring')
    ap.add_argument('--edge_thresh_mm', type=float, default=20.0,
                    help='reject pixels whose local depth neighborhood spans > this (mm)')
    ap.add_argument('--edge_ksize', type=int, default=9,
                    help='neighborhood size (px) for the depth-discontinuity test')
    ap.add_argument('--radius_nb', type=int, default=20,
                    help='radius-outlier: min neighbors within --radius_m to keep a point')
    ap.add_argument('--radius_m', type=float, default=0.012,
                    help='radius-outlier search radius (m)')
    ap.add_argument('--n_frames', type=int, default=1,
                    help='capture+segment N frames with the same clicks (static scene)')
    ap.add_argument('--save_ply', type=int, default=1)
    ap.add_argument('--save_qa', type=int, default=1, help='write color/mask/overlay pngs')
    ap.add_argument('--vis', type=int, default=0, help='pop an Open3D view of each cloud')
    args = ap.parse_args()

    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    pipe, align_filter = open_camera(args.serial, args.color_w, args.color_h,
                                     args.fps, exposure=args.exposure)
    try:
        # let color auto-exposure converge (~1s) before grabbing the prompt frame,
        # otherwise the first frames come out blown-out white
        warmup_autoexposure(pipe, align_filter, max_seconds=args.warmup_seconds)
        color0, fs0 = capture_aligned(pipe, align_filter)
        H, W = color0.shape[:2]

        # robot joint state at capture time, so sim can reset to this exact pose
        robot_state = read_robot_state(args.robot_ip) if args.read_robot else None
        if robot_state is not None:
            json.dump(robot_state,
                      open(os.path.join(args.out_dir, 'robot_state.json'), 'w'), indent=2)

        K, cam = color_intrinsics(pipe)

        # --- prompts ---
        pos = parse_points(args.point)
        neg = parse_points(args.neg)
        if not pos:
            pos, neg_click = pick_points_interactive(color0)
            neg = neg + neg_click
        print(f'[orbbec-seg] prompts: {len(pos)} positive, {len(neg)} negative @ {pos}')

        # --- SAM2 + Orbbec point cloud filter ---
        predictor, device = load_sam2(args.checkpoint, args.model_cfg)
        pcf = make_pointcloud_filter(cam)

        vis = None
        if args.vis:
            vis = o3d.visualization.Visualizer()
            vis.create_window('Orbbec foreground cloud', width=1024, height=768)

        n_written = 0
        for i in range(args.n_frames):
            color, fs = (color0, fs0) if i == 0 else capture_aligned(pipe, align_filter)
            mask = segment(predictor, color, pos, neg, device)
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

            grid = orbbec_pointcloud(pcf, fs)            # (H,W,6) xyz(mm)+rgb
            z_mm = grid[..., 2]

            # foreground = object mask within depth range...
            obj_mask = mask.copy()
            if args.erode_mask > 0:                       # ...minus the silhouette ring
                k = np.ones((2 * args.erode_mask + 1,) * 2, np.uint8)
                obj_mask = cv2.erode(obj_mask.astype(np.uint8), k).astype(bool)
            valid = obj_mask & (z_mm > args.zmin * 1000.0) & (z_mm < args.zmax * 1000.0)
            n_pre = int(valid.sum())
            if args.edge_thresh_mm > 0:                   # ...minus flying pixels on depth steps
                valid &= edge_keep_image(z_mm, args.edge_thresh_mm, args.edge_ksize)
            n_edge = int(valid.sum())

            pts = grid[..., :3][valid] / 1000.0          # mm -> meters
            cols = (grid[..., 3:6][valid] / 255.0).clip(0, 1)

            pcd = toOpen3dCloud(pts.astype(np.float32), cols.astype(np.float32))
            n_radius_in = len(pcd.points)
            pcd = radius_outlier_clean(pcd, args.radius_nb, args.radius_m)
            n_final = len(pcd.points)

            if args.save_ply:
                o3d.io.write_point_cloud(
                    os.path.join(args.out_dir, f'cloud_fg_{i:05d}.ply'), pcd)
            if args.save_qa:
                cv2.imwrite(os.path.join(args.out_dir, f'color_{i:05d}.png'),
                            cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.out_dir, f'mask_{i:05d}.png'),
                            (mask * 255).astype(np.uint8))
                overlay = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                overlay[mask] = (0.5 * overlay[mask]
                                 + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.imwrite(os.path.join(args.out_dir, f'overlay_{i:05d}.png'), overlay)
            if vis is not None:
                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.poll_events(); vis.update_renderer()

            n_written += 1
            print(f'  frame {i:05d}: {int(mask.sum())} masked px '
                  f'-> erode/range {n_pre} -> edge {n_edge} '
                  f'-> radius {n_final} pts')

        if vis is not None:
            print('[orbbec-seg] close the Open3D window to finish...')
            vis.run(); vis.destroy_window()

        json.dump({
            'device': 'Orbbec Femto Bolt', 'serial': args.serial,
            'color_wh': [W, H], 'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
            'cx': float(K[0, 2]), 'cy': float(K[1, 2]),
            'zmin': args.zmin, 'zmax': args.zmax, 'n_frames': n_written,
            'erode_mask': args.erode_mask, 'edge_thresh_mm': args.edge_thresh_mm,
            'edge_ksize': args.edge_ksize, 'radius_nb': args.radius_nb,
            'radius_m': args.radius_m,
            'checkpoint': args.checkpoint, 'model_cfg': args.model_cfg,
            'pos_points': pos, 'neg_points': neg, 'robot_state': robot_state,
            'note': 'foreground-only clouds in the Femto Bolt COLOR frame (meters); '
                    'Orbbec PointCloudFilter cloud masked in-image via SAM2',
        }, open(os.path.join(args.out_dir, 'segment_meta.json'), 'w'), indent=2)
        print(f'[orbbec-seg] wrote {n_written} foreground clouds to {args.out_dir}')
    finally:
        pipe.stop()


if __name__ == '__main__':
    main()
