import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer)
from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper
from diffusion_policy.real_world.ur5e_kinematics import (
    forward_kinematics_calibrated, compute_jacobian_calibrated,
    get_ee_pose, axis_angle_to_quat, quat_to_axis_angle,
    apply_delta_pose, compute_pose_error,
    PAYLOAD_MASS, PAYLOAD_COG,
)


class Command(enum.Enum):
    STOP = 0
    JointTorqueControl = 1   # Joint target -> FK -> OSC torque
    CartesianOSCControl = 2  # Absolute EE target -> direct OSC torque


class RTDEInterpolationController(mp.Process):
    """
    OSC (Operational Space Control) torque controller for UR robot.
    Runs in a separate process to ensure predictable latency.
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 gripper_port=63352,
                 frequency=500,
                 launch_timeout=30,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128,
                 tcp_offset=None,  # [x, y, z, rx, ry, rz] TCP offset from flange
                 # OSC parameters
                 osc_kp_pos=1000.0,
                 osc_kp_rot=50.0,
                 osc_damping_ratio_pos=1.0,
                 osc_damping_ratio_rot=1.0,
                 osc_error_delta_pos=0.05,
                 osc_error_delta_rot=0.3,
                 ):
        """
        Args:
            frequency: Control frequency in Hz (500Hz for UR torque control)
            joints_init: Initial joint positions in radians (6D array)
            joints_init_speed: Speed for initial joint movement (rad/s)
            soft_real_time: Enable round-robin scheduling and real-time priority
            verbose: Print debug messages
            tcp_offset: TCP offset from flange [x, y, z, rx, ry, rz]
            osc_kp_pos: OSC position stiffness
            osc_kp_rot: OSC rotation stiffness
            osc_damping_ratio_pos: OSC position damping ratio
            osc_damping_ratio_rot: OSC rotation damping ratio
        """
        # verify
        assert 0 < frequency <= 500
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEOSCController")
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.tcp_offset = np.array(tcp_offset) if tcp_offset is not None else None
        
        # Torque limits
        self.torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)
        
        # OSC parameters
        self.osc_error_delta_pos = osc_error_delta_pos if osc_error_delta_pos > 0 else None
        self.osc_error_delta_rot = osc_error_delta_rot if osc_error_delta_rot > 0 else None
        stiffness = np.array([osc_kp_pos]*3 + [osc_kp_rot]*3)
        damping_ratio = np.array([osc_damping_ratio_pos]*3 + [osc_damping_ratio_rot]*3)
        self.osc_Kp = np.diag(stiffness)
        self.osc_Kd = np.diag(2 * np.sqrt(stiffness) * damping_ratio)
        
        # build input queue
        example = {
            'cmd': Command.JointTorqueControl.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_ee_pos': np.zeros((3,), dtype=np.float64),
            'target_ee_quat': np.zeros((4,), dtype=np.float64),
            'close_gripper': np.zeros((1,), dtype=np.bool_),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualQ',
                'ActualQd',
                'ActualTCPPose',  # EE pose from robot's FK
                'ActualTCPForce',  # 6D wrench [Fx,Fy,Fz,Tx,Ty,Tz] from built-in F/T sensor
            ]
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())
        example['robot_receive_timestamp'] = time.time()
        example['gripper_current'] = np.zeros(1, dtype=np.float64)
        example['gripper_pos'] = np.zeros(1, dtype=np.float64)
        example['osc_target_pos'] = np.zeros(3, dtype=np.float64)
        example['osc_target_quat'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDETorqueController] Controller process "
                  f"spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_ee_pos': np.zeros((3,), dtype=np.float64),
            'target_ee_quat': np.zeros((4,), dtype=np.float64),
            'close_gripper': np.zeros((1,), dtype=np.bool_),
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def joint_torque_control(self, target_joints, close_gripper):
        """
        Send joint target for OSC torque control.
        FK(target_joints) -> desired EE -> OSC torque.
        
        Args:
            target_joints: Array of 6 target joint positions in radians
            close_gripper: Boolean for gripper state
        """
        assert self.is_alive()
        target_joints = np.array(target_joints)
        assert target_joints.shape == (6,)

        message = {
            'cmd': Command.JointTorqueControl.value,
            'target_joints': target_joints.astype(np.float64),
            'target_ee_pos': np.zeros((3,), dtype=np.float64),
            'target_ee_quat': np.zeros((4,), dtype=np.float64),
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
        }
        self.input_queue.put(message)

    def cartesian_osc_control(self, target_pos, target_quat, close_gripper):
        """
        Send absolute EE target for direct OSC tracking.
        
        Args:
            target_pos: (3,) desired EE position [x, y, z] in base frame
            target_quat: (4,) desired EE orientation as quaternion [w, x, y, z]
            close_gripper: Boolean for gripper state
        """
        assert self.is_alive()
        target_pos = np.array(target_pos, dtype=np.float64)
        target_quat = np.array(target_quat, dtype=np.float64)
        assert target_pos.shape == (3,)
        assert target_quat.shape == (4,)

        message = {
            'cmd': Command.CartesianOSCControl.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
            'target_ee_pos': target_pos,
            'target_ee_quat': target_quat,
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
        }
        self.input_queue.put(message)

    def reset_to_initial_position(self, duration=3.0):
        """
        Reset robot to initial joint position using moveJ.
        
        Args:
            duration: Time to reach initial position (seconds)
        """
        if self.joints_init is not None:
            print(f"Resetting robot to initial position: {self.joints_init}")
            # Note: This requires direct access to rtde_c, so it's handled in run()
            # For now, just set target to initial joints
            self.joint_torque_control(
                target_joints=self.joints_init,
                close_gripper=False
            )
            time.sleep(duration + 0.5)
        else:
            print("Warning: No initial joint positions defined for reset")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def compute_osc_torque(self, target_joints, curr_joints, curr_vel):
        """
        Compute OSC torque from joint targets: FK(target) -> desired EE, then OSC.
        """
        ee_pos_des, ee_quat_des = get_ee_pose(target_joints)
        return self._osc_torque_from_ee(ee_pos_des, ee_quat_des, curr_joints, curr_vel)

    def _osc_torque_from_ee(self, ee_pos_des, ee_quat_des, curr_joints, curr_vel):
        """
        Core OSC torque computation from desired EE pose.
        tau = J^T @ (Kp @ pose_error + Kd @ (-ee_vel))
        """
        # FK for current EE pose (calibrated, REP-103 frame)
        ee_pos_cur, ee_quat_cur = get_ee_pose(curr_joints)
        
        # Jacobian at current config (calibrated, REP-103 frame)
        jacobian = compute_jacobian_calibrated(curr_joints)
        
        # Pose error (6D: position + axis-angle rotation)
        pose_error = compute_pose_error(
            ee_pos_cur, ee_quat_cur, ee_pos_des, ee_quat_des)
        
        # Clip errors to bound maximum task-space force (prevents jamming/spikes)
        if self.osc_error_delta_pos is not None:
            pose_error[:3] = np.clip(pose_error[:3], -self.osc_error_delta_pos, self.osc_error_delta_pos)
        if self.osc_error_delta_rot is not None:
            pose_error[3:] = np.clip(pose_error[3:], -self.osc_error_delta_rot, self.osc_error_delta_rot)
        
        # EE velocity via J @ qdot
        ee_vel = jacobian @ curr_vel
        
        # Task-space PD: F = Kp @ error + Kd @ (-vel)
        des_force = self.osc_Kp @ pose_error + self.osc_Kd @ (-ee_vel)
        
        # Joint torques: tau = J^T @ F
        torque = jacobian.T @ des_force
        
        # Clamp
        torque = np.clip(torque, -self.torque_max, self.torque_max)
        return torque.astype(float)

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start gripper
        gripper = RobotiqGripper()
        gripper.connect(self.robot_ip, self.gripper_port)
        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip, frequency=self.frequency,
                                      flags=RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip, frequency=self.frequency)
        rtde_c.setPayload(PAYLOAD_MASS, PAYLOAD_COG)

        try:
            if self.verbose:
                print(f"[RTDETorqueController] Connect to robot: "
                      f"{robot_ip}")

            # init joints
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init.tolist(),
                                    self.joints_init_speed, 1.4)

            gripper.activate()
            gripper_pos_open = float(gripper.get_open_position())
            gripper_pos_closed = float(gripper.get_closed_position())

            # main loop
            curr_joints = rtde_r.getActualQ()
            current_target_joints = np.array(curr_joints, dtype=np.float64)
            # Cartesian target: initialize from current FK
            init_pos, init_quat = get_ee_pose(np.array(curr_joints))
            current_target_ee_pos = init_pos
            current_target_ee_quat = init_quat
            use_cartesian_target = False
            current_gripper_close = False
            current_gripper_state = 'open'

            iter_idx = 0
            gripper_current_poll_interval = max(1, round(self.frequency / 10))
            last_gripper_current = np.zeros(1, dtype=np.float64)
            last_gripper_pos = np.zeros(1, dtype=np.float64)
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                curr_joints = np.array(rtde_r.getActualQ(), dtype=np.float64)
                curr_vel = np.array(rtde_r.getActualQd(), dtype=np.float64)
                
                # Compute OSC torque command
                if use_cartesian_target and current_target_ee_pos is not None:
                    torque_cmd = self._osc_torque_from_ee(
                        current_target_ee_pos, current_target_ee_quat,
                        curr_joints, curr_vel)
                else:
                    torque_cmd = self.compute_osc_torque(
                        current_target_joints, curr_joints, curr_vel)
                
                # Send torque command
                ok = rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)
                if not ok:
                    if self.verbose:
                        print("[RTDETorqueController] directTorque failed")

                # update gripper state
                if (current_gripper_close and
                        current_gripper_state == 'open'):
                    gripper.move(gripper.get_closed_position(), 250, 128)
                    current_gripper_state = 'closed'
                elif (not current_gripper_close and
                      current_gripper_state == 'closed'):
                    gripper.move(gripper.get_open_position(), 250, 128)
                    current_gripper_state = 'open'

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['robot_receive_timestamp'] = time.time()
                if iter_idx % gripper_current_poll_interval == 0:
                    last_gripper_current = np.array([gripper.get_motor_current()], dtype=np.float64)
                    raw_pos = float(gripper.get_current_position())
                    last_gripper_pos = np.array([(raw_pos - gripper_pos_open) / (gripper_pos_closed - gripper_pos_open)], dtype=np.float64)
                state['gripper_current'] = last_gripper_current
                state['gripper_pos'] = last_gripper_pos
                state['osc_target_pos'] = current_target_ee_pos.copy()
                state['osc_target_quat'] = current_target_ee_quat.copy()
                
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                    commands = None

                # execute commands
                if commands is not None:
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd'][0] if isinstance(command['cmd'], np.ndarray) else command['cmd']

                        if cmd == Command.STOP.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.JointTorqueControl.value:
                            # Joint target -> FK -> OSC
                            current_target_joints = np.array(command['target_joints'], dtype=np.float64)
                            current_target_ee_pos, current_target_ee_quat = get_ee_pose(current_target_joints)
                            use_cartesian_target = False
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            if self.verbose:
                                print("[RTDEOSCController] New joint target: "
                                      f"{current_target_joints}")
                        elif cmd == Command.CartesianOSCControl.value:
                            # Absolute EE target -> direct OSC
                            current_target_ee_pos = np.array(command['target_ee_pos'], dtype=np.float64)
                            current_target_ee_quat = np.array(command['target_ee_quat'], dtype=np.float64)
                            use_cartesian_target = True
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            if self.verbose:
                                print(f"[RTDEOSCController] Cartesian OSC: "
                                      f"target_pos={current_target_ee_pos}")
                        else:
                            keep_running = False
                            break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    freq = 1/(time.perf_counter() - t_start)
                    print(f"[RTDETorqueController] Actual frequency "
                          f"{freq}")

        finally:
            # mandatory cleanup
            try:
                # Send zero torque to stop
                zero_torque = np.zeros(6)
                rtde_c.directTorque(zero_torque.tolist(), friction_comp=False)
                time.sleep(0.1)
                
                # Hold current position briefly to prevent drift
                current_joints = rtde_r.getActualQ()
                rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
                
                # decelerate
                rtde_c.servoStop()
            except Exception as e:
                if self.verbose:
                    print(f"[RTDETorqueController] Cleanup error: {e}")

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDETorqueController] Disconnected from "
                      f"robot: {robot_ip}")
