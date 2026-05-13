# Author: Jimmy Wu
# Date: October 2024

import numpy as np
from cameras import KinovaCamera, LogitechCamera, ThirdPersonCamera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL
from arm_server import ArmManager
from base_server import BaseManager

class ArmOnlyEnv:
    def __init__(self):
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
        self.arm = arm_manager.Arm()
        try:
            self.wrist_camera = KinovaCamera(check_fisheye=False)
        except AssertionError as e:
            print(f'Warning: Could not open wrist camera ({e}). Continuing without it.')
            self.wrist_camera = None
        try:
            self.third_person_camera = ThirdPersonCamera()
        except AssertionError as e:
            print(f'Warning: Could not open third-person camera ({e}). Continuing without it.')
            self.third_person_camera = None

    def get_obs(self):
        obs = {'base_pose': np.zeros(3)}
        obs.update(self.arm.get_state())
        if self.wrist_camera is not None:
            obs['wrist_image'] = self.wrist_camera.get_image()
        if self.third_person_camera is not None:
            obs['third_person_image'] = self.third_person_camera.get_image()
        return obs

    def reset(self):
        print('Resetting arm...')
        self.arm.reset()
        print('Arm has been reset')

    def step(self, action):
        self.arm.execute_action(action)

    def close(self):
        self.arm.close()
        if self.wrist_camera is not None:
            self.wrist_camera.close()
        if self.third_person_camera is not None:
            self.third_person_camera.close()

class BaseOnlyEnv:
    def __init__(self):
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)

    def get_obs(self):
        obs = self.base.get_state()
        obs['base_image'] = self.base_camera.get_image()
        return obs

    def reset(self):
        print('Resetting base...')
        self.base.reset()
        print('Base has been reset')

    def step(self, action):
        self.base.execute_action(action)

    def close(self):
        self.base.close()
        self.base_camera.close()

class RealEnv:
    def __init__(self):
        # RPC server connection for base
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e

        # RPC server connection for arm
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e

        # RPC proxy objects
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()

        # Cameras
        self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
        self.wrist_camera = KinovaCamera()

    def get_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        obs['base_image'] = self.base_camera.get_image()
        obs['wrist_image'] = self.wrist_camera.get_image()
        return obs

    def reset(self):
        print('Resetting base...')
        self.base.reset()

        print('Resetting arm...')
        self.arm.reset()

        print('Robot has been reset')

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.base.execute_action(action)  # Non-blocking
        self.arm.execute_action(action)   # Non-blocking

    def close(self):
        self.base.close()
        self.arm.close()
        self.base_camera.close()
        self.wrist_camera.close()

if __name__ == '__main__':
    import time
    import numpy as np
    from constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
