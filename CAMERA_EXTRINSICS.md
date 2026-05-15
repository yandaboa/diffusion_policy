# Camera Extrinsics for DA3 Pose-Conditioned Depth

Used by `eval_real_robot_depth.py` with `--use_nested --wrist_cam_extrinsic ... --side_cam_extrinsic ...`.

## Overview

DA3NESTED-GIANT-LARGE accepts per-frame world-to-camera matrices (4×4 float64) for each
view.  Two calibration files are needed — one per camera.  Without them, the model runs
in multi-view mode without pose conditioning (still better than single-frame).

---

## `wrist_cam_in_ee.npy` — camera-in-wrist_3_link

**Shape:** (4, 4) float64

**Meaning:** The pose of the wrist (RealSense D415) camera expressed in the
`wrist_3_link` frame.  A point `p` in camera coordinates maps to wrist_3_link
coordinates via `T @ [p; 1]`.

```
T_wrist←cam = [ R_cam_in_wrist | t_cam_origin_in_wrist ]
              [ 0   0   0      | 1                     ]
```

**From Isaac Lab** — if the wrist camera is a child of `wrist_3_link` with:
```python
OffsetCfg(pos=(px, py, pz), rot=(qw, qx, qy, qz))
```
then build the matrix as:
```python
import numpy as np
from scipy.spatial.transform import Rotation

pos = np.array([px, py, pz])          # camera origin in wrist_3_link frame
R   = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: xyzw order

T = np.eye(4, dtype=np.float64)
T[:3, :3] = R
T[:3,  3] = pos
np.save('wrist_cam_in_ee.npy', T)
```

**Pass to eval:**
```
--wrist_cam_extrinsic wrist_cam_in_ee.npy
```

---

## `side_cam_w2c.npy` — world-to-side-camera

**Shape:** (4, 4) float64

**Meaning:** Transforms a 3D point from the robot base frame into the side
(RealSense D435) camera frame.  "World" = robot base frame.

```
T_cam←base = [ R_base_to_cam | -R_base_to_cam @ t_cam_in_base ]
             [ 0   0   0     |  1                              ]
```

**From Isaac Lab** — if the side camera is fixed in the world with pose
(position and orientation expressed in the robot base frame):
```python
import numpy as np
from scipy.spatial.transform import Rotation

pos_in_base = np.array([px, py, pz])           # camera origin in base frame
R_in_base   = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

T_cam_in_base = np.eye(4, dtype=np.float64)   # camera-to-world (need to invert)
T_cam_in_base[:3, :3] = R_in_base
T_cam_in_base[:3,  3] = pos_in_base

side_cam_w2c = np.linalg.inv(T_cam_in_base)   # world-to-camera (what DA3 needs)
np.save('side_cam_w2c.npy', side_cam_w2c)
```

If your sim already stores the side camera as a "robot base to camera" transform
(i.e. it already maps base-frame points into camera-frame points), save it directly
without inverting.

**Pass to eval:**
```
--side_cam_extrinsic side_cam_w2c.npy
```

---

## Sanity checks

```python
import numpy as np

wrist_cam_T  = np.load('wrist_cam_in_ee.npy')
side_cam_w2c = np.load('side_cam_w2c.npy')

# Wrist cam: z-axis of camera in wrist_3_link frame (should point roughly forward)
print("wrist cam z-axis in wrist frame:", wrist_cam_T[:3, 2])

# Side cam: robot base origin mapped into side camera frame
# (should be a ~1 m away vector with negative z — camera is looking toward origin)
origin_in_side_cam = side_cam_w2c @ np.array([0., 0., 0., 1.])
print("base origin in side cam frame:", origin_in_side_cam[:3])
```

---

## Full eval command with pose conditioning

```bash
python eval_real_robot_depth.py \
    -i <policy.pt> \
    -o <output_dir> \
    --robot_ip <ip> \
    --use_nested \
    --da3_only \
    --da3_process_res 378 \
    --save_video \
    --wrist_cam_extrinsic wrist_cam_in_ee.npy \
    --side_cam_extrinsic  side_cam_w2c.npy
```

Without the two `--*_extrinsic` flags, the model runs multi-view without pose
conditioning (automatic fallback, no error).

---

## Camera serials (for reference)

| Camera | Serial | Role |
|--------|--------|------|
| D455 (front) | 215122255213 | RGB only, not used for depth policy |
| D435 (side)  | 832112070487 | Fixed scene view — depth + RGB |
| D415 (wrist) | 746112060198 | Robot-mounted — depth + RGB |
