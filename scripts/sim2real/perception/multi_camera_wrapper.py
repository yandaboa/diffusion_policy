import cv2
import matplotlib.pyplot as plt
import numpy as np


class MultiCameraWrapper:

    def __init__(
        self,
        rgb=True,
        depth=False,
        ir=False,
        high_res_rgb=False,
        align=None,
        type="realsense",
    ):
        """
        Multi Camera Wrapper that returns images in order of serial number
        params:
        specific_cameras: list of camera objects
        use_threads: bool, whether to use threads for reading the cameras
        type: str, type of camera to use. Options are 'realsense' and 'zed'
        :num_expected_cameras: int, number of cameras expected. If none, won't have default cameras
        """
        self.rgb = rgb
        self.depth = depth
        self.ir = ir
        self.high_res_rgb = high_res_rgb
        self.align = align

        self.type = type
        self._reset_cameras()
        self.num_cameras = len(self._all_cameras)

    def _reset_cameras(self):
        """
        Resets the cameras to the state they were in when the object was first created
        """
        if hasattr(self, "_all_cameras"):
            for camera in self._all_cameras:
                try:
                    camera.disable_camera()
                except RuntimeError as e:
                    # This means that the camera is not plugged in right now
                    continue

        self._all_cameras = []

        if self.type == "realsense":
            from perception.realsense import gather_realsense_cameras

            self._all_cameras.extend(
                gather_realsense_cameras(
                    rgb=self.rgb,
                    depth=self.depth,
                    ir=self.ir,
                    high_res_rgb=self.high_res_rgb,
                    align=self.align,
                )
            )
        elif self.type == "orbbec":
            from perception.orbbec import gather_orbbec_cameras

            self._all_cameras.extend(
                gather_orbbec_cameras(
                    rgb=self.rgb,
                    depth=self.depth,
                    ir=self.ir,
                    high_res_rgb=self.high_res_rgb,
                    align=self.align,
                )
            )
        # elif self.type == "zed":
        #     from perception.zed_camera import gather_zed_cameras
        #     self._all_cameras.extend(gather_zed_cameras())
        else:
            raise NotImplementedError(
                "Type of camera not implemented. You requested no specific cameras with no type"
            )

        assert (
            hasattr(self, "num_cameras") or len(self._all_cameras) > 0
        ), "No cameras found"
        self._sort_cams_by_serial()

    def read_cameras(self):
        try:
            all_frames = []
            for camera in self._all_cameras:
                curr_feed = camera.read_camera()
                if curr_feed is not None:
                    all_frames.append(curr_feed)
            # if len(all_frames) != self.num_cameras * 2:
            #     raise RuntimeError(
            #         f"Number of cameras changed, expected {self.num_cameras}, got {len(all_frames)//2}"
            #     )
            return all_frames

        except RuntimeError as e:
            print(f"Error reading cameras: {e}")
            input("Press enter when cameras are fixed:\t")
            self._reset_cameras()
            return self.read_cameras()

    def disable_cameras(self):
        for camera in self._all_cameras:
            camera.disable_camera()

    def _sort_cams_by_serial(self):
        """
        Sorts the cameras by serial number
        """
        self._all_cameras.sort(key=lambda x: x._serial_number)

    # https://stackoverflow.com/a/76802895
    def _estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            # SOLVEPNP_SQPNP, SOLVEPNP_IPPE_SQUARE
            # nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            nada, R, t = cv2.solvePnP(
                marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash

    def get_aruco_poses(
        self, marker_size=0.15, aruco_dict=None, steps=1, verbose=False
    ):
        if aruco_dict is None:
            aruco_dict = cv2.aruco.DICT_6X6_50

        tvecs = []
        rotation_matrices = []
        for camera in self._all_cameras:
            tvec, rotation_matrix = self._get_aruco_pose(
                camera,
                marker_size=marker_size,
                aruco_dict=aruco_dict,
                steps=steps,
                verbose=verbose,
            )
            tvecs.append(tvec)
            rotation_matrices.append(rotation_matrix)
        return tvecs, rotation_matrices

    def _get_aruco_pose(
        self,
        camera,
        marker_size=0.15,
        aruco_dict=None,
        steps=1,
        verbose=False,
    ):

        if aruco_dict is None:
            aruco_dict = cv2.aruco.DICT_6X6_50

        frame = camera.read_camera()["rgb"]

        intrinsics = camera.calibration["intrinsics"]
        camera_matrix = intrinsics["rgb"]["cameraMatrix"]
        dist_coeff = intrinsics["rgb"]["distCoeffs"]

        # # create aruco detector
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        # same as: (opencv-contrib-python 4.6.0.66)
        # dictionary = cv2.aruco.Dictionary_get(aruco_dict)
        # parameters = cv2.aruco.DetectorParameters_create()

        # run detection
        for _ in range(steps):

            frame = camera.read_camera()["rgb"]

            # detect markers in frame
            corners, ids, _ = detector.detectMarkers(frame)
            # same as: (opencv-contrib-python 4.6.0.66)
            # corners, ids, rejected_candidates = cv2.aruco.detectMarkers(
            #             frame, dictionary=dictionary, parameters=parameters
            #         )

            if len(corners) == 0:
                print("WARNING: no aruco tag detected!")
                return None, None

            if ids is not None:

                # check first marker (assuming there only is one)
                rvec, tvec, _ = self._estimatePoseSingleMarkers(
                    corners[0],
                    marker_size=marker_size,
                    mtx=camera_matrix,
                    distortion=dist_coeff,
                )
                # same as: (opencv-contrib-python 4.6.0.66)
                # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                #     corners[0],
                #     marker_size,
                #     camera_matrix,
                #     dist_coeff,
                # )

                # plot marker orientation
                if verbose:
                    img_rend = frame.copy()
                    img_rend = cv2.aruco.drawDetectedMarkers(img_rend, corners, ids)
                    length_of_axis = 0.1
                    # x: red, y: green, z: blue
                    img_rend = cv2.drawFrameAxes(
                        img_rend,
                        camera_matrix,
                        dist_coeff,
                        rvec[0],
                        tvec[0],
                        length_of_axis,
                    )
                    plt.imshow(img_rend)

                # rodrigues rotation to matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])

        return tvec[0], rotation_matrix
