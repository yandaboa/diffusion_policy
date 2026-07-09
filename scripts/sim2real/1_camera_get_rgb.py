import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from perception.multi_camera_wrapper import MultiCameraWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["realsense", "orbbec"], default="realsense",
                        help="Camera backend to use.")
    args = parser.parse_args()

    # Initialize camera
    multi_camera_wrapper = MultiCameraWrapper(rgb=True, depth=False, ir=False, high_res_rgb=False, align="rgb", type=args.camera)
    
    # Get frames from first camera
    camera = multi_camera_wrapper._all_cameras[0]
    frames = camera.read_camera()
    rgb = frames["rgb"]

    # Display and save the image
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()
    
    # Save the image
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception/calibrations/real_rgb.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, rgb)
    print(f"Saved RGB image to {save_path}") 