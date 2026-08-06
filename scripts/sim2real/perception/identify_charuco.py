"""Identify the ArUco dictionary of an unknown ChArUco / ArUco board.

You know the markers are 6x6 bits and there are 60 of them, which means the
dictionary size must be >= 60 -> one of DICT_6X6_100 / _250 / _1000 (DICT_6X6_50
is too small). The bit pattern for a given ID differs between these, so the only
way to tell is to detect: the correct dictionary decodes the most markers with
IDs in [0, 60).

Usage:
    # from an image file
    python identify_charuco.py --image board.jpg

    # grab a frame live from the camera and test that
    python identify_charuco.py --camera realsense
    python identify_charuco.py --camera orbbec
"""
import argparse
import cv2
import numpy as np

# An 11x11 ChArUco has 60 markers, so the dictionary size must be >= 60 (the _50
# variants can't hold it). We still list _50 to catch a mis-spec, but exhaustively
# sweep every bit-size family (4x4..7x7) x size (100/250/1000) since we're not
# sure the markers are 6x6. Predefined dicts are NOT nested, so the same ID has a
# different pattern in each -> only the true dictionary decodes cleanly.
CANDIDATES = {}
for _bits in ("4X4", "5X5", "6X6", "7X7"):
    for _n in (50, 100, 250, 1000):
        _name = f"DICT_{_bits}_{_n}"
        CANDIDATES[_name] = getattr(cv2.aruco, _name)
CANDIDATES["DICT_APRILTAG_36h11"] = cv2.aruco.DICT_APRILTAG_36h11


def detect(gray, dict_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return 0, None
    return len(ids), ids.flatten()


def grab_from_camera(kind):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from perception.multi_camera_wrapper import MultiCameraWrapper
    mcw = MultiCameraWrapper(rgb=True, depth=False, ir=False,
                             high_res_rgb=False, align="rgb", type=kind)
    cam = mcw._all_cameras[0]
    rgb = cam.read_camera()["rgb"]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to a photo of the board")
    ap.add_argument("--camera", choices=["realsense", "orbbec"], default=None)
    ap.add_argument("--save", default=None, help="optional: write annotated image here")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read image: {args.image}")
    elif args.camera:
        img = grab_from_camera(args.camera)
    else:
        raise SystemExit("pass --image PATH or --camera {realsense,orbbec}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"\nimage: {img.shape[1]}x{img.shape[0]}\n")
    print(f"{'dictionary':<22} {'#detected':>9}   id range")
    print("-" * 55)
    results = []
    for name, did in CANDIDATES.items():
        n, ids = detect(gray, did)
        rng = f"{ids.min()}..{ids.max()}" if n else "-"
        results.append((n, name, did, ids))
        print(f"{name:<22} {n:>9}   {rng}")

    results.sort(reverse=True, key=lambda r: r[0])
    best_n, best_name, best_did, best_ids = results[0]
    print("-" * 55)
    if best_n == 0:
        print("\nNo markers detected in ANY dictionary. Get a sharper / closer "
              "photo of the board filling the frame, even lighting, no glare.")
        return

    print(f"\n==> Best match: {best_name}  ({best_n} markers detected)")
    if best_ids.max() < 60:
        print(f"    All IDs in [0,60) as expected for a 60-marker board. "
              f"High confidence this is your dictionary.")
    else:
        print(f"    NOTE: detected IDs go up to {best_ids.max()} (>=60). "
              f"Either it's a larger board or a few false detections.")

    if args.save:
        aruco_dict = cv2.aruco.getPredefinedDictionary(best_did)
        det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        cv2.imwrite(args.save, img)
        print(f"    annotated image -> {args.save}")


if __name__ == "__main__":
    main()
