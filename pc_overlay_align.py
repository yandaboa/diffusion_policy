"""Live sim<->real point-cloud alignment (base-frame aligner).

The point-cloud analogue of sim_overlay_align.py. Instead of blending a sim RGB
render over the live feed, this renders a *simulation* point cloud (in the sim
robot-base frame) as a fixed reference, and overlays the *live Orbbec* point
cloud transformed into the same world by the ChArUco-board extrinsic. The two
won't line up perfectly (the ChArUco board corner frame != the sim base frame),
so you nudge a rigid correction by hand until the real cloud lands on the sim
cloud. The correction you dial in IS the transform between the real (ChArUco)
base and the sim base.

What's rendered:
  * sim cloud   -- fixed, solid orange, in the sim base frame (the world).
  * real cloud  -- live Orbbec RGBD backprojected to camera frame, then mapped to
                   world by  T_total = T_corr @ T_charuco.  T_charuco is the fixed
                   board->world calibration; T_corr is what you move by hand.
  * captured    -- optional static green cloud (--captured) previously captured by this
                   rig in the base frame, to check it isn't biased/shifted vs sim or live.
  * world axes  -- at the sim-base origin (== ChArUco board top-left corner).

The view itself is a normal Open3D scene: drag to orbit, right-drag / scroll to
pan / zoom. The keyboard moves the *real cloud's extrinsic*, not the view:

  TRANSLATE (world axes)        ROTATE (about world origin)
    d / a   +X / -X               l / j   +yaw / -yaw   (Z)
    w / s   +Y / -Y               i / k   +pitch/-pitch (Y)
    e / c   +Z / -Z               o / u   +roll / -roll (X)

  QUICK FLIP 180 deg            STEP SIZE
    7  about X                    =  / -   pos step x2 / /2
    8  about Y                    ]  / [   rot step x2 / /2
    9  about Z

  f  freeze/unfreeze live grab        v  real color: true RGB <-> solid cyan
  m  toggle real render: points <-> surface mesh
  b  toggle sim cloud   n  toggle real cloud   x  toggle crop to sim extent
  g  toggle captured-cloud overlay (--captured, green)
  r  reset correction to identity (back to raw ChArUco)
  p  print + save current extrinsic / correction to JSON
  ESC or close window  quit

Run (foundstereo env owns pyorbbecsdk):
    conda run -n foundstereo python pc_overlay_align.py \
        --sim_pc /home/yandabao/UWLab-patrick-private/scene_capture_front_orbbec_idx3_pc_base.npy \
        --calib scripts/sim2real/perception/calibrations/most_recent_charuco_calib.json

Resume a prior alignment and keep refining (p re-saves a new JSON each time):
    ... pc_overlay_align.py --sim_pc ... --resume Log/pc_align/pc_align_1782366735.json
"""
import os
import sys
import json
import time
import argparse
import threading

import numpy as np
import open3d as o3d

# Reuse the existing Orbbec backend (aligned depth + RGB + color intrinsics).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts", "sim2real"))

# Canonical "latest hand-aligned extrinsic" the eval script reads to project real
# points into the sim base frame. Every 'p'-save overwrites this in addition to the
# timestamped history file under --out_dir.
CANONICAL_EXTRINSIC = os.path.join(
    _HERE, "scripts", "sim2real", "perception", "calibrations",
    "most_recent_hand_aligned_extrinsic.json")


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------
def rot_axis(axis, ang):
    """3x3 rotation of `ang` rad about world 'x'|'y'|'z'."""
    c, s = np.cos(ang), np.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def homog(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def mat_to_quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def load_calib(path, serial=None):
    data = json.load(open(path))
    if isinstance(data, dict):
        data = [data]
    cam = None
    if serial is not None:
        cam = next((c for c in data if str(c.get("camera_serial_number")) == str(serial)), None)
    if cam is None:
        cam = data[0]
    T = np.array(cam["extrinsics_raw"], dtype=np.float64)  # camera -> world (board corner)
    K = np.array(cam["intrinsics_raw"], dtype=np.float64)
    return T, K, cam.get("camera_serial_number")


def load_overlay_cloud(path):
    """Load a previously captured cloud (.ply / .npz / .npy) -> (pts (N,3), frame).

    A perception_test .npz also carries a 'frame' tag ('base'/'camera'); we surface it so
    we can warn if a camera-frame capture is overlaid on the base-frame world. PLY/NPY have
    no frame metadata (assumed base).
    """
    p = str(path)
    if p.endswith(".ply") or p.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(p)
        return np.asarray(pcd.points, dtype=np.float64), None
    if p.endswith(".npz"):
        d = np.load(p)
        pts = np.asarray(d["points"], dtype=np.float64)[:, :3]
        frame = str(d["frame"]) if "frame" in d.files else None
        return pts, frame
    a = np.load(p)
    return np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])[:, :3], None


# ---------------------------------------------------------------------------
# aligner
# ---------------------------------------------------------------------------
class PCAligner:
    def __init__(self, args):
        self.args = args

        # --- sim reference cloud (world / sim base frame) ---
        sim = np.load(args.sim_pc)
        sim = sim.reshape(-1, sim.shape[-1])[:, :3].astype(np.float64)
        self.sim_pts = sim
        self.sim_aabb = (sim.min(0) - args.crop_margin, sim.max(0) + args.crop_margin)
        print(f"[pc-align] sim cloud: {len(sim)} pts, "
              f"x{sim[:,0].min():.2f}..{sim[:,0].max():.2f} "
              f"y{sim[:,1].min():.2f}..{sim[:,1].max():.2f} "
              f"z{sim[:,2].min():.2f}..{sim[:,2].max():.2f}")

        # --- ChArUco extrinsic (camera -> board world) + optional resume state ---
        resume = json.load(open(args.resume)) if args.resume else None
        if args.calib:
            self.T_charuco, self.K_calib, serial = load_calib(args.calib, args.serial)
        elif resume is not None:
            self.T_charuco = np.array(resume["T_charuco_cam"], dtype=np.float64)
            self.K_calib, serial = None, "(from resume)"
        else:
            raise SystemExit("need --calib (or --resume, which carries its own calib)")
        print(f"[pc-align] loaded extrinsic for camera {serial}\n{self.T_charuco.round(4)}")

        # --- live camera backend (orbbec front / realsense side) ---
        self.cam = self._open_camera(args.camera, args.serial)
        self.serial = str(self.cam._serial_number)
        K = np.asarray(self.cam.calibration["intrinsics"]["rgb"]["cameraMatrix"], float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        print(f"[pc-align] live {args.camera} {self.serial}")
        if serial is not None and str(serial) != self.serial:
            print(f"[pc-align] WARNING: calib is for serial {serial} but the live camera is "
                  f"{self.serial} -- pass --serial so the extrinsic matches the physical camera.")

        # background grabber: keep the camera serviced at full rate off the render
        # thread (a slow/blocked render must never starve the Orbbec pipeline), and
        # do the expensive SW align + backprojection here, not in the draw loop.
        self._lock = threading.Lock()
        self._latest = None        # (pts_cam (N,3) f64, col (N,3) f64)
        self._grabber = None

        # --- adjustable correction (world->world): sim_base = T_corr @ charuco_world ---
        # Resume a prior alignment if given, else start from identity (raw ChArUco).
        self.T_corr = np.array(resume["T_corr"], dtype=np.float64) if resume else np.eye(4)
        self.pos_step = args.pos_step
        self.rot_step = np.deg2rad(args.rot_step_deg)
        if resume is not None:
            if args.calib and not np.allclose(self.T_charuco,
                                              np.array(resume["T_charuco_cam"], float), atol=1e-6):
                print("[pc-align] WARNING: --calib differs from the calibration saved in "
                      "--resume; applying the saved correction on top of the NEW calib.")
            # self.pos_step = float(resume.get("pos_step_m", self.pos_step))
            # self.rot_step = np.deg2rad(float(resume.get("rot_step_deg", np.rad2deg(self.rot_step))))
            print(f"[pc-align] resumed correction from {args.resume}\n{self.T_corr.round(4)}")
        self.T_corr0 = self.T_corr.copy()   # 'r' resets to this session's start

        # --- runtime state ---
        self.frozen = False
        self.real_rgb = True
        self.show_sim = True
        self.show_real = True
        self.show_cap = True
        self.crop = args.crop
        self.render_mode = args.render   # 'points' | 'mesh'
        self._cur = None                 # latest payload consumed by the render loop
        self._sim_dirty = False
        self._cap_dirty = False
        self.running = True

        # --- geometries ---
        self.sim_pcd = o3d.geometry.PointCloud()
        self.sim_pcd.points = o3d.utility.Vector3dVector(self.sim_pts)
        self.sim_pcd.paint_uniform_color([1.0, 0.55, 0.0])  # orange
        self.real_pcd = o3d.geometry.PointCloud()
        self.real_mesh = o3d.geometry.TriangleMesh()
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

        # --- optional captured-cloud overlay (e.g. pc_debug/perception_test.ply) ---
        # A static reference in the SAME base/world frame as the sim cloud, so you can
        # check the points this rig captures aren't biased/shifted vs sim or the live feed.
        self.cap_pcd = None
        if args.captured:
            cpts, cframe = load_overlay_cloud(args.captured)
            if cframe == "camera":
                print("[pc-align] WARNING: captured cloud is in CAMERA frame, not base -- it will "
                      "NOT line up with the base-frame world. Re-save with --perception_frame base.")
            self.cap_pts = cpts
            self.cap_pcd = o3d.geometry.PointCloud()
            self.cap_pcd.points = o3d.utility.Vector3dVector(cpts)
            self.cap_pcd.paint_uniform_color([0.1, 1.0, 0.2])  # green
            print(f"[pc-align] captured overlay: {len(cpts)} pts from {args.captured}"
                  + (f" (frame={cframe})" if cframe else ""))

    # -- open the requested live backend, picking a specific camera by serial --
    def _open_camera(self, kind, serial):
        """Return one camera whose read_camera() yields color-aligned {rgb, depth}.

        Both backends expose the same dict + color intrinsics, so _backproject works
        unchanged. RealSense depth is warped onto the color grid with align='color'
        (the analogue of the Orbbec 'rgb' SW-align), so it shares the color intrinsics.
        With >1 camera on a backend (2 RealSenses: side D435 + wrist D415), --serial is
        required so we don't grab the wrong one.
        """
        if kind == "orbbec":
            from perception.orbbec import gather_orbbec_cameras
            cams = gather_orbbec_cameras(rgb=True, depth=True, align="rgb")
            if not cams:
                raise RuntimeError("No orbbec camera found.")
            found = [str(c._serial_number) for c in cams]
            if serial is not None:
                match = [c for c in cams if str(c._serial_number) == str(serial)]
                if not match:
                    raise SystemExit(f"--serial {serial} not among connected orbbec cameras {found}")
                return match[0]
            if len(cams) > 1:
                raise SystemExit(f"{len(cams)} orbbec cameras {found}; pass --serial to pick one.")
            return cams[0]

        if kind == "realsense":
            # Open ONLY the requested device. gather_realsense_cameras() starts a
            # depth+color pipeline on EVERY RealSense, so the wrist D415's depth stream
            # runs alongside the side D435's -- two 640x480 depth streams can exceed USB
            # bandwidth and stall delivery ("Frame didn't arrive"). Starting just the one
            # we need avoids that (and leaves the wrist camera untouched).
            import pyrealsense2 as rs
            from perception.realsense import RealSenseCamera

            def find_dev():
                devs = list(rs.context().devices)
                found = [str(d.get_info(rs.camera_info.serial_number)) for d in devs]
                if not devs:
                    raise RuntimeError("No realsense camera found.")
                if serial is not None:
                    d = next((dv for dv, s in zip(devs, found) if s == str(serial)), None)
                    if d is None:
                        raise SystemExit(f"--serial {serial} not among connected realsense cameras {found}")
                    return d
                if len(devs) > 1:
                    raise SystemExit(f"{len(devs)} realsense cameras {found}; pass --serial to pick one "
                                     f"(e.g. the side D435, not the wrist D415).")
                return devs[0]

            dev = find_dev()
            cam = RealSenseCamera(dev, rgb=True, depth=True, align="color")
            if self._warmup_realsense(cam) == 0:      # hard stall -> hardware_reset + retry once
                print("[pc-align] RealSense delivered no frames; hardware_reset + retry...")
                try:
                    cam.disable_camera()
                except Exception:
                    pass
                dev.hardware_reset()
                time.sleep(5.0)
                cam = RealSenseCamera(find_dev(), rgb=True, depth=True, align="color")
                if self._warmup_realsense(cam) == 0:
                    raise SystemExit("still no frames after reset -- unplug/replug the camera and retry.")
            return cam

        raise SystemExit(f"unknown --camera {kind!r}")

    @staticmethod
    def _warmup_realsense(cam, want=10, budget_s=8.0):
        """Drop initial frames so the stream settles; returns how many arrived.

        Right after start (especially following an unclean Ctrl-C of a prior run) a D4xx
        can stall for seconds -- a single 5s wait_for_frames then times out and the session
        spins on 'Frame didn't arrive'. Poll with short timeouts, tolerating misses.
        """
        t0, got = time.time(), 0
        while time.time() - t0 < budget_s and got < want:
            try:
                cam._pipeline.wait_for_frames(1000)
                got += 1
            except Exception:
                pass
        if got:
            print(f"[pc-align] RealSense warmup: {got} frames, stream settled ({time.time() - t0:.1f}s)")
        return got

    # -- backproject an aligned RGBD frame to a camera-frame cloud --
    def _backproject(self, fr):
        rgb, depth = fr["rgb"], fr["depth"]
        H, W = depth.shape
        st = self.args.stride
        vs, us = np.mgrid[0:H:st, 0:W:st]
        z = depth[vs, us].astype(np.float32) / 1000.0
        m = (z > self.args.zmin) & (z < self.args.zmax)
        z, us, vs = z[m], us[m], vs[m]
        x = (us - self.cx) * z / self.fx
        y = (vs - self.cy) * z / self.fy
        pts = np.stack([x, y, z], 1).astype(np.float64)
        col = rgb[vs, us].astype(np.float64) / 255.0
        return pts, col

    # -- triangulate the organized depth map into a camera-frame surface mesh --
    def _build_mesh_cam(self, fr):
        rgb, depth = fr["rgb"], fr["depth"]
        st = self.args.stride
        z = depth[::st, ::st].astype(np.float32) / 1000.0  # (R, C) organized grid
        R, C = z.shape
        us = np.arange(depth.shape[1])[::st][:C]
        vs = np.arange(depth.shape[0])[::st][:R]
        uu, vv = np.meshgrid(us, vs)
        valid = (z > self.args.zmin) & (z < self.args.zmax)
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        verts = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float64)
        col = rgb[vv, uu].reshape(-1, 3).astype(np.float64) / 255.0
        # two triangles per grid quad, dropping quads with an invalid corner or a
        # depth jump > mesh_thresh (so we don't web over object/background edges).
        idx = np.arange(R * C).reshape(R, C)
        v00, v10 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
        v01, v11 = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
        z00, z10 = z[:-1, :-1].ravel(), z[1:, :-1].ravel()
        z01, z11 = z[:-1, 1:].ravel(), z[1:, 1:].ravel()
        vq = (valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]).ravel()
        dz = np.maximum.reduce([np.abs(z00 - z11), np.abs(z10 - z01),
                                np.abs(z00 - z10), np.abs(z00 - z01)])
        good = vq & (dz < self.args.mesh_thresh)
        t1 = np.stack([v00[good], v10[good], v11[good]], -1)
        t2 = np.stack([v00[good], v11[good], v01[good]], -1)
        tris = np.concatenate([t1, t2], 0).astype(np.int32)
        return verts, col, tris

    def _grab_payload(self, fr):
        if self.render_mode == "mesh":
            return ("mesh",) + self._build_mesh_cam(fr)
        return ("points",) + self._backproject(fr)

    def grab_real(self):
        """Blocking single grab (used to prime before the thread starts)."""
        with self._lock:
            self._latest = self._grab_payload(self.cam.read_camera())

    def _grab_loop(self):
        """Continuously service the camera so the pipeline never stalls."""
        while self.running:
            if self.frozen:
                time.sleep(0.02)
                continue
            try:
                payload = self._grab_payload(self.cam.read_camera())
            except Exception as e:
                print(f"[pc-align] grab error: {e}")
                continue
            with self._lock:
                self._latest = payload

    def total(self):
        return self.T_corr @ self.T_charuco

    def _empty(self, geom, attr):
        if len(getattr(geom, attr)):
            geom.clear()

    # -- recompute world geometry from the latest payload and push to the GPU --
    def refresh_real(self):
        cur = self._cur
        if cur is None or not self.show_real:
            self._empty(self.real_pcd, "points")
            self._empty(self.real_mesh, "vertices")
            return
        T = self.total()
        Rm, t = T[:3, :3], T[:3, 3]
        if cur[0] == "mesh":
            self._empty(self.real_pcd, "points")
            _, verts, col, tris = cur
            world = verts @ Rm.T + t
            if not self.real_rgb:
                col = np.tile([0.0, 0.85, 1.0], (len(world), 1))
            self.real_mesh.vertices = o3d.utility.Vector3dVector(world)
            self.real_mesh.triangles = o3d.utility.Vector3iVector(tris)
            self.real_mesh.vertex_colors = o3d.utility.Vector3dVector(col)
            self.real_mesh.compute_vertex_normals()  # for proper shading
            return
        # points
        self._empty(self.real_mesh, "vertices")
        _, pts, col = cur
        world = pts @ Rm.T + t
        if self.crop:
            lo, hi = self.sim_aabb
            keep = np.all((world >= lo) & (world <= hi), axis=1)
            world, col = world[keep], col[keep]
        n = len(world)
        if n == 0:
            return
        # Resample to a FIXED count so the legacy Visualizer reuses its GPU buffers
        # instead of reallocating every frame (the live valid-depth count wobbles by
        # a few dozen points each frame, which otherwise forces a realloc). linspace
        # keeps the subsample spatially uniform and stable frame-to-frame.
        idx = np.linspace(0, n - 1, self.args.max_points).astype(np.int64)
        world = world[idx]
        col = col[idx] if self.real_rgb else None
        self.real_pcd.points = o3d.utility.Vector3dVector(world)
        if col is None:
            col = np.tile([0.0, 0.85, 1.0], (len(world), 1))  # solid cyan
        self.real_pcd.colors = o3d.utility.Vector3dVector(col)

    def refresh_sim(self):
        """Rebuild the (static) sim cloud only when its visibility toggles."""
        pts = self.sim_pts if self.show_sim else np.empty((0, 3))
        self.sim_pcd.points = o3d.utility.Vector3dVector(pts)
        self.sim_pcd.paint_uniform_color([1.0, 0.55, 0.0])
        self._sim_dirty = False

    def refresh_cap(self):
        """Rebuild the (static) captured-cloud overlay only when its visibility toggles."""
        pts = self.cap_pts if self.show_cap else np.empty((0, 3))
        self.cap_pcd.points = o3d.utility.Vector3dVector(pts)
        self.cap_pcd.paint_uniform_color([0.1, 1.0, 0.2])
        self._cap_dirty = False

    # -- keyboard handlers (each returns False; we re-render in the main loop) --
    def _translate(self, axis_vec):
        self.T_corr = homog(np.eye(3), np.asarray(axis_vec) * self.pos_step) @ self.T_corr

    def _rotate(self, axis, sign):
        R = rot_axis(axis, sign * self.rot_step)
        self.T_corr = homog(R, [0, 0, 0]) @ self.T_corr

    def _flip(self, axis):
        self.T_corr = homog(rot_axis(axis, np.pi), [0, 0, 0]) @ self.T_corr
        print(f"[pc-align] flipped 180deg about world {axis.upper()}")

    def register(self, vis):
        def K(ch):  # key code for a character
            return ord(ch.upper())

        # translation
        vis.register_key_callback(K("d"), lambda v: self._translate([1, 0, 0]) or False)
        vis.register_key_callback(K("a"), lambda v: self._translate([-1, 0, 0]) or False)
        vis.register_key_callback(K("w"), lambda v: self._translate([0, 1, 0]) or False)
        vis.register_key_callback(K("s"), lambda v: self._translate([0, -1, 0]) or False)
        vis.register_key_callback(K("e"), lambda v: self._translate([0, 0, 1]) or False)
        vis.register_key_callback(K("c"), lambda v: self._translate([0, 0, -1]) or False)
        # rotation
        vis.register_key_callback(K("l"), lambda v: self._rotate("z", +1) or False)
        vis.register_key_callback(K("j"), lambda v: self._rotate("z", -1) or False)
        vis.register_key_callback(K("i"), lambda v: self._rotate("y", +1) or False)
        vis.register_key_callback(K("k"), lambda v: self._rotate("y", -1) or False)
        vis.register_key_callback(K("o"), lambda v: self._rotate("x", +1) or False)
        vis.register_key_callback(K("u"), lambda v: self._rotate("x", -1) or False)
        # flips
        vis.register_key_callback(K("7"), lambda v: self._flip("x") or False)
        vis.register_key_callback(K("8"), lambda v: self._flip("y") or False)
        vis.register_key_callback(K("9"), lambda v: self._flip("z") or False)
        # step size
        vis.register_key_callback(K("="), lambda v: self._step(pos=2.0))
        vis.register_key_callback(K("-"), lambda v: self._step(pos=0.5))
        vis.register_key_callback(K("]"), lambda v: self._step(rot=2.0))
        vis.register_key_callback(K("["), lambda v: self._step(rot=0.5))
        # toggles / actions
        vis.register_key_callback(K("f"), lambda v: self._toggle("frozen"))
        vis.register_key_callback(K("v"), lambda v: self._toggle("real_rgb"))
        vis.register_key_callback(K("b"), lambda v: self._toggle("show_sim"))
        vis.register_key_callback(K("n"), lambda v: self._toggle("show_real"))
        vis.register_key_callback(K("g"), lambda v: self._toggle("show_cap"))
        vis.register_key_callback(K("x"), lambda v: self._toggle("crop"))
        vis.register_key_callback(K("m"), lambda v: self._toggle_mesh())
        vis.register_key_callback(K("r"), lambda v: self._reset())
        vis.register_key_callback(K("p"), lambda v: self._save())
        vis.register_key_callback(256, lambda v: self._quit())  # ESC

    def _step(self, pos=None, rot=None):
        if pos:
            self.pos_step *= pos
        if rot:
            self.rot_step *= rot
        print(f"[pc-align] pos_step={self.pos_step*1000:.2f}mm  rot_step={np.rad2deg(self.rot_step):.2f}deg")
        return False

    def _toggle(self, name):
        setattr(self, name, not getattr(self, name))
        if name == "show_sim":
            self._sim_dirty = True
        elif name == "show_cap":
            self._cap_dirty = True
        print(f"[pc-align] {name} = {getattr(self, name)}")
        return False

    def _toggle_mesh(self):
        self.render_mode = "mesh" if self.render_mode == "points" else "points"
        self._cur = None  # drop the stale-mode payload; next grab refills
        print(f"[pc-align] render_mode = {self.render_mode}")
        return False

    def _reset(self):
        self.T_corr = self.T_corr0.copy()
        print("[pc-align] correction reset to session start "
              f"({'resumed' if np.any(self.T_corr0 != np.eye(4)) else 'identity / raw ChArUco'})")
        return False

    def _quit(self):
        self.running = False
        return False

    def _save(self):
        T_total = self.total()
        R, t = T_total[:3, :3], T_total[:3, 3]
        out = {
            "camera_serial_number": self.serial,            # which physical camera this is
            "camera_kind": self.args.camera,                # orbbec | realsense
            "T_charuco_cam": self.T_charuco.tolist(),       # input calibration (cam->board)
            "T_corr": self.T_corr.tolist(),                 # hand correction (board->sim base)
            "T_total_cam_simbase": T_total.tolist(),        # final cam->sim-base extrinsic
            "camera_base_pos": t.tolist(),
            "camera_base_quat_wxyz": mat_to_quat(R).tolist(),
            "pos_step_m": self.pos_step,
            "rot_step_deg": float(np.rad2deg(self.rot_step)),
        }
        # 1) timestamped history file (for resuming / rollback)
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, f"pc_align_{int(time.time())}.json")
        json.dump(out, open(path, "w"), indent=2)
        # 2) canonical "latest" file the eval script reads to project into the base frame
        os.makedirs(os.path.dirname(CANONICAL_EXTRINSIC), exist_ok=True)
        json.dump(out, open(CANONICAL_EXTRINSIC, "w"), indent=2)
        # 3) per-camera file, keyed by serial, so aligning a second camera never clobbers
        #    the first (the canonical file above only holds whichever was aligned last).
        per_serial = os.path.join(os.path.dirname(CANONICAL_EXTRINSIC),
                                  f"most_recent_hand_aligned_extrinsic_{self.serial}.json")
        json.dump(out, open(per_serial, "w"), indent=2)
        print("\n" + "=" * 64)
        print("Final camera->sim-base extrinsic (T_total):")
        print(np.array(T_total).round(5))
        print(f"  pos (x,y,z) = {t.round(5)}")
        print(f"  quat wxyz   = {mat_to_quat(R).round(6)}")
        print("Hand correction (charuco-base -> sim-base, T_corr):")
        print(np.array(self.T_corr).round(5))
        print(f"saved -> {path}")
        print(f"saved -> {CANONICAL_EXTRINSIC}  (canonical 'latest' for eval)")
        print(f"saved -> {per_serial}  (per-camera, keyed by serial)")
        print("=" * 64 + "\n")
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("sim/real point-cloud align", width=1280, height=800)
        opt = vis.get_render_option()
        opt.point_size = float(self.args.point_size)
        opt.background_color = np.array([0.05, 0.05, 0.05])

        # prime one real frame so the initial view is framed, then add geometry once
        self._sim_dirty = False
        try:
            self.grab_real()
        except Exception as e:
            print(f"[pc-align] initial grab failed ({e}); continuing")
        with self._lock:
            self._cur = self._latest
        self.refresh_sim()
        self.refresh_real()
        vis.add_geometry(self.sim_pcd)   # static: added once, only touched on toggle
        vis.add_geometry(self.real_pcd)
        vis.add_geometry(self.real_mesh)
        if self.cap_pcd is not None:     # static captured-cloud overlay (green)
            self.refresh_cap()
            vis.add_geometry(self.cap_pcd)
        vis.add_geometry(self.axes)
        self.register(vis)

        # start the background grabber (camera serviced independently of render rate)
        self._grabber = threading.Thread(target=self._grab_loop, daemon=True)
        self._grabber.start()

        print(__doc__[__doc__.index("What's rendered"):__doc__.index("Run (")])
        while self.running:
            # pull the most recent payload the grabber produced (no camera I/O here)
            with self._lock:
                self._cur = self._latest
            self.refresh_real()
            vis.update_geometry(self.real_pcd)
            vis.update_geometry(self.real_mesh)
            if self._sim_dirty:          # only when 'b' toggled sim visibility
                self.refresh_sim()
                vis.update_geometry(self.sim_pcd)
            if self.cap_pcd is not None and self._cap_dirty:  # 'g' toggled captured cloud
                self.refresh_cap()
                vis.update_geometry(self.cap_pcd)
            if not vis.poll_events():
                break
            vis.update_renderer()

        self.running = False
        if self._grabber is not None:
            self._grabber.join(timeout=1.0)
        vis.destroy_window()
        try:
            self.cam.disable_camera()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Live sim<->real point-cloud base-frame aligner.")
    ap.add_argument("--sim_pc", required=True, help="Sim point cloud .npy (N,3) in sim base frame.")
    ap.add_argument("--calib", default=None, help="ChArUco calibration JSON (camera->board extrinsic).")
    ap.add_argument("--resume", default=None,
                    help="A prior Log/pc_align/*.json to continue aligning from (loads T_corr; "
                         "carries its own calib if --calib is omitted).")
    ap.add_argument("--camera", choices=["orbbec", "realsense"], default="orbbec",
                    help="Live backend: orbbec (front) or realsense (side D435). Default orbbec.")
    ap.add_argument("--serial", default=None,
                    help="Camera serial: picks BOTH the calib-JSON entry and the physical "
                         "camera. Required for realsense when 2 are connected (side D435 vs wrist D415).")
    ap.add_argument("--captured", default=None,
                    help="Optional captured cloud (.ply/.npz/.npy) to overlay as a static green "
                         "reference, e.g. pc_debug/perception_test.ply. Must be in the base frame "
                         "(perception_test default) to line up. Toggle with 'g'.")
    ap.add_argument("--render", choices=["points", "mesh"], default="points",
                    help="Live render style ('m' toggles at runtime).")
    ap.add_argument("--mesh_thresh", type=float, default=0.03,
                    help="Max depth jump (m) across a mesh quad before it's left open (edges).")
    ap.add_argument("--stride", type=int, default=3, help="Pixel stride when backprojecting the live cloud.")
    ap.add_argument("--max_points", type=int, default=80000,
                    help="Fixed live-cloud point count (keeps GPU buffers stable; lower = faster).")
    ap.add_argument("--zmin", type=float, default=0.1, help="Min valid depth (m).")
    ap.add_argument("--zmax", type=float, default=3.0, help="Max valid depth (m).")
    ap.add_argument("--pos_step", type=float, default=0.002, help="Initial translation step (m).")
    ap.add_argument("--rot_step_deg", type=float, default=2.0, help="Initial rotation step (deg).")
    ap.add_argument("--crop", action="store_true", help="Start with real cloud cropped to sim extent.")
    ap.add_argument("--crop_margin", type=float, default=0.3, help="Margin (m) around sim AABB when cropping.")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--out_dir", default="Log/pc_align", help="Where 'p' saves the alignment JSON.")
    args = ap.parse_args()
    PCAligner(args).run()


if __name__ == "__main__":
    main()
