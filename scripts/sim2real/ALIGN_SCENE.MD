First, capture a picture of the sim scene:
```
python scripts_v2/tools/sim2real/scene_capture.py --enable_cameras --camera front_camera --idx 3
```
Then, create a live overlay so you can reset the scene exactly the same
```
python sim_overlay_align.py sim.png \
    --robot_ip 192.168.1.10 \
    --joints <j1>,<j2>,<j3>,<j4>,<j5>,<j6>
```

Actually forget the above. To align point clouds, record a point cloud from the robot in sim with:
```
python scripts_v2/tools/sim2real/scene_capture.py     --enable_cameras --camera orbbec --idx 3     --save_pc
```
^ make sure to choose a idx (initial configuration) where you can see the robot clearly in real, so in the middle of the scene.

Then, in real, place the charuco board on the table with the top left corner underneath the center of the ur5 meta support (the grey metal base). After setting up the scene, run this to grab the initial extrinsic as a starting point.
```
python scripts/sim2real/0_camera_calibrate.py --camera orbbec --board charuco
```
Then move the robots to the same joint position as the reset state we recorded points from. I just do this in a jank way, I run this script that isn't really meant for it (of course swap out the joint positions for the ones displayed by step 1, scene_capture.py):
```
python sim_overlay_align.py /home/yandabao/UWLab-patrick-private/scene_capture_front_orbbec_idx3.png --camera orbbec     --robot_ip 192.168.1.10     --joints 0.363852,-1.338813,2.095192,-2.831392,-1.785279,0.331381
```
Finally, with the scene aligned physically and a decent estimate of extrinsics, start interactively align the point clouds:
```
python pc_overlay_align.py   --sim_pc /home/yandabao/UWLab-patrick-private/scene_capture_front_orbbec_idx3_pc_base.npy   --calib /home/yandabao/diffusion_policy/scripts/sim2real/perception/calibrations/most_recent_charuco_calib.json
```
or to resume if we already have an extrinsic:
```
python pc_overlay_align.py   --sim_pc /home/yandabao/UWLab-patrick-private/scene_capture_front_orbbec_idx3_pc_base.npy   --resume /home/yandabao/diffusion_policy/scripts/sim2real/perception/calibrations/most_recent_hand_aligned_extrinsic.json
```
