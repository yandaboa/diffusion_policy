# Orbbec Femto Bolt point-cloud tools

A small toolchain for capturing, cropping, saving, and drift-checking point
clouds from the **Orbbec Femto Bolt** RGB-D camera. The flow is: pick a table
plane interactively → stream a live cropped cloud → save a single cropped cloud +
the robot's joint pose → later reset the robot and re-capture to check whether the
camera has moved.

All scripts run in the **`foundstereo`** conda env (it has `pyorbbecsdk2`,
`open3d`, `cv2`, and `ur_rtde`). Invoke python by its full path:

```bash
~/miniforge3/envs/foundstereo/bin/python <script>.py ...
```

> **Camera is single-owner.** Only one process can hold the Femto Bolt at a time.
> If a script dies with `uvc_open failed ... Return Code: -6`, something else owns
> it — usually OrbbecViewer or a crashed run. Fix: `pkill -f OrbbecViewer` (or kill
> the stale python) and re-run.

---

## Files at a glance

| File | Role |
|---|---|
| `orbbec_camera.py` | **Core API.** Talking to the camera: open/warmup/capture/intrinsics/point-cloud. No perception, no post-processing. |
| `interactive_point_cloud.py` | Open a `.ply` in an Open3D window; Shift+click to read off 3D point coordinates (used to pick the 3 plane points). |
| `point_distances.py` | Given 3 coplanar points, report the in-plane frame + pairwise in-plane displacements. |
| `orbbec_live_crop.py` | **Main tool.** Live cropped point-cloud viewer (plane-aligned rectangular prism + floor removal). Also a one-shot save mode (cloud + robot joints). |
| `orbbec_compare_capture.py` | Reset the robot to a saved pose, re-capture, and compare against the saved cloud — a **camera-drift check**. |
| `orbbec_segment_pointclouds.py` | (Separate) SAM2 foreground segmentation on the live cloud; also built on `orbbec_camera`. |

The dependency graph:

```
orbbec_camera.py  (core: pyorbbecsdk + numpy)
        ├── orbbec_live_crop.py ──────┐
        │        (crop_cloud, plane_frame, ... reused)
        ├── orbbec_compare_capture.py ┘  (+ ur_rtde for moveJ, open3d ICP)
        └── orbbec_segment_pointclouds.py  (+ sam2, torch)
```

---

## `orbbec_camera.py` — core camera API

The only module that imports `pyorbbecsdk`. Everything else builds on it.

| Function | Purpose |
|---|---|
| `open_camera(serial, color_w, color_h, fps, exposure=None)` | Start color+depth, software-aligned to color (Femto Bolt has no HW D2C). Returns `(pipe, align_filter)`. |
| `warmup_autoexposure(pipe, align_filter, ...)` | Discard frames until color AE settles (the first ~1 s is blown out). |
| `capture_aligned(pipe, align_filter, ...)` | Grab one synced frame → `(color_rgb HxWx3 uint8, aligned_frameset)`. |
| `color_intrinsics(pipe)` | `(K 3x3, OBCameraParam)` for the color stream. |
| `make_pointcloud_filter(cam)` | Build a `PointCloudFilter` emitting RGB points. |
| `orbbec_pointcloud(pcf, frameset)` | Dense `(H, W, 6)` grid `[x,y,z (mm), r,g,b]`, row-major over the color frame. |

Units: positions are in **millimeters**, camera frame. Downstream code converts
to meters.

---

## Step 1 — pick the plane: `interactive_point_cloud.py` + `point_distances.py`

Open a saved `.ply` and pick 3 points that lie on the table/floor plane:

```bash
~/miniforge3/envs/foundstereo/bin/python interactive_point_cloud.py /path/to/cloud.ply
```

- **Shift + Left click** = pick a point (prints index + XYZ, in mm).
- **Shift + Right click** = undo. Press `Q` / close to finish.

These 3 points (call them `p0, p1, p2`) define the crop plane used everywhere
below. `p1` is the crop **center**; `p0→p1` sets the in-plane X axis.

`point_distances.py` is a helper to sanity-check the picked points — it builds the
in-plane frame and reports pairwise in-plane displacements:

```bash
~/miniforge3/envs/foundstereo/bin/python point_distances.py "x0,y0,z0" "x1,y1,z1" "x2,y2,z2"
```

---

## Step 2 — live cropped viewer / one-shot save: `orbbec_live_crop.py`

Streams the Bolt and keeps only the points inside a **plane-aligned rectangular
prism** centered on `p1`, then removes the floor.

### The crop, conceptually

The plane frame (right-handed, unit vectors):

```
x_axis = normalize(p1 - p0)              # in-plane
z_axis = normalize((p1-p0) × (p2-p0))    # plane normal
y_axis = z_axis × x_axis                 # in-plane
```

A point `P` (with `d = P - p1`) is kept when:

```
x_min ≤ d·x_axis ≤ x_max          (in-plane bounds, asymmetric)
y_min ≤ d·y_axis ≤ y_max
```

The prism is **infinite along the normal** — height is handled by floor removal,
not by the crop. `--rotate_deg` spins the in-plane axes about the normal so the
rectangle lines up with a physical edge.

### Floor removal

- **Default (on):** one-sided height crop — keep only points **more than
  `--floor_height_m` above** the plane. This drops the floor slab *and* everything
  below the plane (`w ≤ 0`).
- **`--slope_floor` (opt-in):** slope-aware region grow from the 3 reference
  points across the per-pixel height field. A neighbor joins the floor only if its
  height delta to the *adjacent* floor pixel is `< --plane_slope_m` **and** its
  absolute height stays within `--floor_height_m`. Absorbs a gently sloped floor
  while keeping objects that rise off it with a sharp edge. Everything under the
  plane is also dropped.
- **`--flip_normal`:** which side is "up" depends on the `p0/p1/p2` order. If the
  crop keeps the wrong side (or comes up empty), add this flag.

### Viewer reference geometry

- **Amber rectangle** = the in-plane crop window (at height 0).
- **Coordinate triad at p1**: **red = x_axis, green = y_axis, blue = z_axis (normal)**.

### Key flags (current defaults)

| Flag | Default | Meaning |
|---|---|---|
| `--p0 / --p1 / --p2` | picked points (mm) | plane points; `p1` = crop center |
| `--x_min / --x_max` | `-0.425 / 0.5` | in-plane X bounds from p1 (m) |
| `--y_min / --y_max` | `-0.425 / 0.45` | in-plane Y bounds from p1 (m) |
| `--rotate_deg` | `25.0` | spin in-plane axes about the normal (deg) |
| `--floor_height_m` | `0.034` | keep only points this far above the plane; `0` disables |
| `--flip_normal` | off | flip which side counts as "up" |
| `--slope_floor` | off | use slope-aware region grow instead of the plain band |
| `--plane_slope_m` | `0.001` | (slope) max per-pixel height delta (m) |
| `--seed_win` | `12` | (slope) seed search radius around each reference pixel (px) |
| `--voxel_m` | `0.0` | voxel-downsample the output (m); `0` = full density |
| `--exposure` | auto | fix color exposure (disables AE) |
| `--show_box` | `1` | draw crop rectangle + axis triad |

### Run it — live viewer

```bash
~/miniforge3/envs/foundstereo/bin/python orbbec_live_crop.py
# tune the window, then maybe try slope-aware floor:
~/miniforge3/envs/foundstereo/bin/python orbbec_live_crop.py --slope_floor
# if the floor side is inverted:
~/miniforge3/envs/foundstereo/bin/python orbbec_live_crop.py --flip_normal
```

Close the window (or Ctrl-C) to stop. A per-30-frame point count prints to the
terminal so you can tell if the crop is eating too much / too little.

### Run it — one-shot save (`--save_dir`)

Captures **one** frame, applies the identical crop, and writes two files to the
directory, then exits (no viewer):

```bash
~/miniforge3/envs/foundstereo/bin/python orbbec_live_crop.py \
    --save_dir tmp/capture0 --robot_ip 192.168.1.10
```

Output in `tmp/capture0/`:

| File | Contents |
|---|---|
| `cloud_crop.ply` | the cropped cloud (meters, camera frame) |
| `robot_state.json` | UR5e joints (`joints_rad/deg`, `tcp_pose`) **+** all crop/plane params + point count |

Robot read is best-effort (`--read_robot 0` to skip); if `ur_rtde` is missing or
the arm is unreachable it warns and writes `"robot_state": null`.

> Save mode has no viewer, so you can't eyeball `--flip_normal`. Do a quick live
> run first to confirm the crop looks right, then re-run with the same flags +
> `--save_dir`.

---

## Step 3 — camera-drift check: `orbbec_compare_capture.py`

The inverse of save mode. Loads a saved dataset, **resets the robot to the saved
joint pose**, re-captures with the identical crop, and compares the new cloud
against the reference to decide whether the **camera has moved**.

```bash
# reset robot, recapture, compare, and show an overlay:
~/miniforge3/envs/foundstereo/bin/python orbbec_compare_capture.py \
    --dataset_dir tmp/capture0 --vis

# arm already in place / don't move it:
~/miniforge3/envs/foundstereo/bin/python orbbec_compare_capture.py \
    --dataset_dir tmp/capture0 --no_move
```

### How it decides

- **Direct nearest-neighbor distances** (no alignment) — how far apart the two
  clouds sit as-is.
- **Rigid ICP** (new → reference) — the best-fit transform; its translation /
  rotation magnitude is what the camera appears to have **moved by**, and the
  inlier RMSE / fitness say how clean the fit is.

| Situation | Signal | Verdict |
|---|---|---|
| camera unmoved, scene same | near-identity ICP, low RMSE | `camera steady` |
| camera moved, scene same | large ICP transform, low RMSE | `CAMERA MOVED` |
| scene changed / low overlap | poor fit (high RMSE) | `INCONCLUSIVE` |

Outputs go to `<dataset_dir>/compare/`: `captured_crop.ply` + `compare_report.json`.
With `--vis`: **gray = reference, red = new (as-is), green = new ICP-aligned.**

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--dataset_dir` | (required) | dir with `cloud_crop.ply` + `robot_state.json` |
| `--no_move` | off | skip the robot reset (arm already positioned) |
| `--yes` | off | don't prompt before moving the robot |
| `--robot_ip` | from JSON | UR5e IP override |
| `--speed / --accel` | `0.5 / 0.5` | moveJ joint speed / accel (rad/s, rad/s²) |
| `--settle_s` | `0.5` | pause after the move before capturing |
| `--icp_max_corr_m` | `0.05` | ICP max correspondence distance (m) |
| `--icp_voxel_m` | `0.005` | voxel size for the ICP solve only (m) |
| `--move_thresh_mm` | `5.0` | ICP translation above this flags a move |
| `--rot_thresh_deg` | `1.0` | ICP rotation above this flags a move |
| `--rmse_thresh_mm` | `8.0` | inlier RMSE below this = "clean fit" |
| `--vis` | off | show reference / new / aligned overlay |

> **Safety:** the robot move is gated behind a `[y/N]` confirmation (skip with
> `--yes`), uses gentle defaults, and prints the largest per-joint motion before
> asking. `--no_move` skips it entirely.

> **Key assumption:** the physical scene inside the crop must be **unchanged**
> between the two captures — otherwise the difference reflects the scene, not the
> camera. That's exactly why the robot is reset (so the arm isn't a scene change).

### Caveat: crop params not stored

Save mode does **not** record `plane_slope_m`, `seed_win`, or `voxel_m` in
`robot_state.json`. The compare script falls back to `orbbec_live_crop`'s defaults
and exposes CLI overrides for them. If you routinely tune those, store them in the
JSON so the round-trip is fully faithful.

---

## Typical end-to-end session

```bash
PY=~/miniforge3/envs/foundstereo/bin/python

# 1. pick the 3 plane points from a saved cloud
$PY interactive_point_cloud.py /path/to/some_cloud.ply
#    -> note p0, p1, p2 (mm); paste into orbbec_live_crop defaults or pass --p0/--p1/--p2

# 2. dial in the crop live
$PY orbbec_live_crop.py            # adjust --x_*/--y_*/--rotate_deg/--floor_height_m/--flip_normal

# 3. save a reference capture (cloud + robot pose)
$PY orbbec_live_crop.py --save_dir tmp/capture0 --robot_ip 192.168.1.10

# ... time passes; suspect the camera may have been bumped ...

# 4. reset the robot, re-capture, and check for drift
$PY orbbec_compare_capture.py --dataset_dir tmp/capture0 --vis
```
