# Author: Jimmy Wu
# Date: October 2024
#
# Note: This code is only intended for debugging the base and arm controllers.
# In external code, avoid directly importing Vehicle as done here.
# Instead, please use the RPC server in base_server.py, which runs the
# low-level controller in a dedicated, real-time process to help minimize
# unintended latency spikes caused by external code.
#
# Gamepad button/axis mapping (Logitech F710, XInput mode):
#   Buttons: A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7
#   Axes:    Left X=0, Left Y=1, Right X=3, Right Y=4
#   Hat:     D-pad (hat 0): up=(0,1), down=(0,-1)
#
# GamepadTeleop (standalone base velocity control):
#   Start/Back -> enable/disable, hold LB or RB to move
#   Left stick -> X/Y velocity, Right stick X -> angular velocity
#   LB = local frame, RB = global frame
#
# GamepadArmTeleop (standalone arm control):
#   Start -> reset arm (home + open gripper)
#   Hold LB or RB (dead-man's switch) to move arm:
#     Left stick X/Y -> EE X/Y translation
#     D-pad up/down  -> EE +Z/-Z translation
#     Right stick X  -> EE roll rotation
#     Right stick Y  -> EE pitch rotation
#     X button       -> EE +yaw rotation
#     Y button       -> EE -yaw rotation
#   Hold A -> close gripper, hold B -> open gripper
#
# GamepadArmPolicy / GamepadBasePolicy (episode recording via main.py):
#   Start -> begin episode, Back -> end episode
#   A -> save episode, B -> discard episode
#   Start (after episode ended) -> reset env

import argparse
import os
import signal
import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from scipy.spatial.transform import Rotation as R
from arm_server import ArmManager
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY

pygame.init()

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

def _gamepad_save_prompt(joy, writer):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False
    print('Save episode? Press A (yes) or B (no)')
    prev_a = joy.get_button(0)
    prev_b = joy.get_button(1)
    while True:
        pygame.event.pump()
        a = joy.get_button(0)
        b = joy.get_button(1)
        if a and not prev_a:
            return True
        if b and not prev_b:
            print('Discarding episode')
            return False
        prev_a, prev_b = a, b
        time.sleep(0.05)

class GamepadTeleop:
    def __init__(self):
        self.joy = Joystick(0)  # Logitech F710
        self.vehicle = None

    def run(self):
        from base_controller import Vehicle
        last_enabled = False
        frame = None
        print('Press the "Start" button on the gamepad to start control')
        while True:
            pygame.event.pump()

            # Start control
            if not self.vehicle and self.joy.get_button(7):  # 7 is the "Start" button
                self.vehicle = Vehicle(max_vel=(1.0, 1.0, 3.14), max_accel=(0.5, 0.5, 2.36))
                self.vehicle.start_control()
                last_enabled = False
                frame = 'local'
                print('Control started')

            # Stop control
            if self.vehicle and self.joy.get_button(6):  # 6 is the "Back" button
                self.vehicle.stop_control()
                self.vehicle = None
                print('Control stopped')

            if self.vehicle:
                # Hold down left/right bumper to enable control in local/global frame
                left_bumper = self.joy.get_button(4)
                right_bumper = self.joy.get_button(5)
                frame = 'local' if left_bumper else 'global'
                if left_bumper or right_bumper:
                    if not last_enabled:
                        print(f'Robot enabled ({frame} frame)')
                        last_enabled = True

                    # Compute unscaled target velocity
                    x = -self.joy.get_axis(1)  # Left analog stick
                    y = -self.joy.get_axis(0)  # Left analog stick
                    th = -self.joy.get_axis(3)  # Right analog stick
                    target_velocity = np.array([x, y, th])

                    # Apply deadzone for joystick drift
                    target_velocity = apply_deadzone(target_velocity)

                    # Send command to robot
                    target_velocity = self.vehicle.max_vel * target_velocity
                    self.vehicle.set_target_velocity(target_velocity, frame=frame)

                elif last_enabled:
                    print('Robot disabled')
                    last_enabled = False

            time.sleep(0.01)

class GamepadArmPolicy:
    ARM_MAX_POS_DELTA = np.array([0.02, 0.02, 0.02])  # meters per step at 10 Hz
    ARM_MAX_ROT_DELTA = 0.05  # radians per step at 10 Hz
    ARM_GRIPPER_DELTA = 0.1

    def __init__(self, print_instructions=True):
        self.joy = Joystick(0)
        self._target_pos = None
        self._target_quat = None
        self._gripper_pos = 0.0
        self._last_enabled = False
        self._prev_start = False
        self._prev_back = False
        self._episode_ended = False
        if print_instructions:
            print('=== Kinova Arm Gamepad Teleop ===')
            print()
            print('Episode control:')
            print('  Start (button 7) -> begin episode')
            print('  Back  (button 6) -> end episode')
            print('  A     (button 0) -> save episode')
            print('  B     (button 1) -> discard episode')
            print()
            print('Translation (hold LB or RB to enable):')
            print('  Left stick X/Y -> EE X/Y translation')
            print('  D-pad up/down  -> EE +Z/-Z translation')
            print()
            print('Rotation (hold LB or RB to enable):')
            print('  Right stick X  -> roll')
            print('  Right stick Y  -> pitch')
            print('  X button       -> +yaw')
            print('  Y button       -> -yaw')
            print()
            print('Gripper (hold LB or RB to enable):')
            print('  Hold A         -> close gripper')
            print('  Hold B         -> open gripper')
            print()
            print(f'(Gamepad: {self.joy.get_name()}, {self.joy.get_numaxes()} axes, '
                  f'{self.joy.get_numbuttons()} buttons, {self.joy.get_numhats()} hats)')

    def seed(self, pos, quat, gripper_pos=0.0):
        self._target_pos = pos.copy()
        self._target_quat = quat.copy()
        self._gripper_pos = gripper_pos
        self._last_enabled = False

    def reset(self):
        self._target_pos = None
        self._target_quat = None
        self._gripper_pos = 0.0
        self._last_enabled = False
        self._prev_start = False
        self._prev_back = False
        self._episode_ended = False
        print('Press Start on gamepad when ready to begin episode')
        while True:
            pygame.event.pump()
            start = self.joy.get_button(7)
            if start and not self._prev_start:
                self._prev_start = start
                break
            self._prev_start = start
            time.sleep(0.05)

    def _read_action(self, obs):
        if self._target_pos is None:
            self._target_pos = obs['arm_pos'].copy()
            self._target_quat = obs['arm_quat'].copy()
            self._gripper_pos = float(obs['gripper_pos'][0])

        left_bumper = self.joy.get_button(4)
        right_bumper = self.joy.get_button(5)
        if not (left_bumper or right_bumper):
            if self._last_enabled:
                print('Control disabled')
                self._last_enabled = False
            return None

        if not self._last_enabled:
            print('Control enabled')
            self._last_enabled = True

        # Gripper: hold A to close, hold B to open
        a_button = self.joy.get_button(0)
        b_button = self.joy.get_button(1)
        self._gripper_pos = float(np.clip(
            self._gripper_pos + self.ARM_GRIPPER_DELTA * (float(a_button) - float(b_button)),
            0.0, 1.0,
        ))

        # Translation: accumulate deltas onto local target (avoids feedback oscillation)
        dx = -self.joy.get_axis(1)   # left stick Y axis -> X (forward/back)
        dy = -self.joy.get_axis(0)   # left stick X axis -> Y (left/right)
        dz = float(self.joy.get_hat(0)[1])  # D-pad up=+1, down=-1
        pos_deltas = apply_deadzone(np.array([dx, dy]))
        self._target_pos = self._target_pos + self.ARM_MAX_POS_DELTA * np.array([pos_deltas[0], pos_deltas[1], dz])

        # Rotation: right stick -> roll/pitch, X/Y buttons -> +/-yaw
        droll  = -self.joy.get_axis(3)
        dpitch = self.joy.get_axis(4)
        dyaw   = float(self.joy.get_button(2)) - float(self.joy.get_button(3))
        rot_deltas = apply_deadzone(np.array([droll, dpitch]), deadzone_size=0.1)
        delta_rot = R.from_euler('ZYX', [
            self.ARM_MAX_ROT_DELTA * dyaw,
            self.ARM_MAX_ROT_DELTA * rot_deltas[1],
            self.ARM_MAX_ROT_DELTA * rot_deltas[0],
        ])
        self._target_quat = (delta_rot * R.from_quat(self._target_quat)).as_quat()
        if self._target_quat[3] < 0:
            np.negative(self._target_quat, out=self._target_quat)

        return {
            'arm_pos': self._target_pos,
            'arm_quat': self._target_quat,
            'gripper_pos': np.array([self._gripper_pos]),
        }

    def step(self, obs):
        pygame.event.pump()
        start = self.joy.get_button(7)
        back = self.joy.get_button(6)

        if self._episode_ended:
            if start and not self._prev_start:
                self._prev_start = start
                self._prev_back = back
                return 'reset_env'
            self._prev_start = start
            self._prev_back = back
            return None

        if back and not self._prev_back:
            self._episode_ended = True
            self._prev_start = start
            self._prev_back = back
            return 'end_episode'

        self._prev_start = start
        self._prev_back = back
        return self._read_action(obs)

    def save_prompt(self, writer):
        return _gamepad_save_prompt(self.joy, writer)


class GamepadArmTeleop:
    def __init__(self):
        self.arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            self.arm_manager.connect()
        except Exception as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
        self.arm = self.arm_manager.Arm()
        self._policy = GamepadArmPolicy(print_instructions=False)

    def run(self):
        joy = self._policy.joy
        print('=== Kinova Arm Gamepad Teleop ===')
        print()
        print('Setup:')
        print('  Start          -> reset arm (home position + open gripper)')
        print()
        print('Translation (hold LB or RB to enable):')
        print('  Left stick X/Y -> EE X/Y translation')
        print('  D-pad up/down  -> EE +Z/-Z translation')
        print()
        print('Rotation (hold LB or RB to enable):')
        print('  Right stick X  -> roll')
        print('  Right stick Y  -> pitch')
        print('  X button       -> +yaw')
        print('  Y button       -> -yaw')
        print()
        print('Gripper (hold LB or RB to enable):')
        print('  Hold A         -> close gripper')
        print('  Hold B         -> open gripper')
        print()
        print('Press Start to begin.')
        print(f'(Gamepad: {joy.get_name()}, {joy.get_numaxes()} axes, '
              f'{joy.get_numbuttons()} buttons, {joy.get_numhats()} hats)')

        arm_ready = False
        prev_start = False
        while True:
            pygame.event.pump()
            start = joy.get_button(7)
            if start and not prev_start:
                print('Resetting arm...')
                self.arm.reset()
                state = self.arm.get_state()
                self._policy.seed(state['arm_pos'], state['arm_quat'])
                arm_ready = True
                print('Arm ready')
            prev_start = start

            if not arm_ready:
                time.sleep(0.1)
                continue

            obs = self.arm.get_state()
            action = self._policy._read_action(obs)
            if action is not None:
                self.arm.execute_action(action)

            time.sleep(0.1)


class GamepadBasePolicy:
    BASE_MAX_POS_DELTA = np.array([0.05, 0.05])  # meters per step at 10 Hz
    BASE_MAX_THETA_DELTA = 0.15  # radians per step at 10 Hz

    def __init__(self, print_instructions=True):
        self.joy = Joystick(0)
        self._target_pose = None  # [x, y, theta] in global frame
        self._last_enabled = False
        self._prev_start = False
        self._prev_back = False
        self._episode_ended = False
        if print_instructions:
            print('=== Mobile Base Gamepad Teleop ===')
            print()
            print('Episode control:')
            print('  Start (button 7) -> begin episode')
            print('  Back  (button 6) -> end episode')
            print('  A     (button 0) -> save episode')
            print('  B     (button 1) -> discard episode')
            print()
            print('Movement (hold LB for local frame, RB for global frame):')
            print('  Left stick X/Y -> X/Y movement')
            print('  Right stick X  -> rotation')
            print()
            print(f'(Gamepad: {self.joy.get_name()}, {self.joy.get_numaxes()} axes, '
                  f'{self.joy.get_numbuttons()} buttons, {self.joy.get_numhats()} hats)')

    def seed(self, pose):
        self._target_pose = pose.copy()
        self._last_enabled = False

    def reset(self):
        self._target_pose = None
        self._last_enabled = False
        self._prev_start = False
        self._prev_back = False
        self._episode_ended = False
        print('Press Start on gamepad when ready to begin episode')
        while True:
            pygame.event.pump()
            start = self.joy.get_button(7)
            if start and not self._prev_start:
                self._prev_start = start
                break
            self._prev_start = start
            time.sleep(0.05)

    def _read_action(self, obs):
        if self._target_pose is None:
            self._target_pose = obs['base_pose'].copy()

        left_bumper = self.joy.get_button(4)
        right_bumper = self.joy.get_button(5)
        if not (left_bumper or right_bumper):
            if self._last_enabled:
                print('Control disabled')
                self._last_enabled = False
            return None

        frame = 'local' if left_bumper else 'global'
        if not self._last_enabled:
            print(f'Control enabled ({frame} frame)')
            self._last_enabled = True

        x = -self.joy.get_axis(1)   # left stick Y axis
        y = -self.joy.get_axis(0)   # left stick X axis
        th = -self.joy.get_axis(3)  # right stick X axis
        velocity = apply_deadzone(np.array([x, y, th]))

        # Transform joystick input from robot-local frame to global frame
        if frame == 'local':
            theta = obs['base_pose'][2]
            cos_th, sin_th = np.cos(theta), np.sin(theta)
            dx = self.BASE_MAX_POS_DELTA[0] * (cos_th * velocity[0] - sin_th * velocity[1])
            dy = self.BASE_MAX_POS_DELTA[1] * (sin_th * velocity[0] + cos_th * velocity[1])
        else:
            dx = self.BASE_MAX_POS_DELTA[0] * velocity[0]
            dy = self.BASE_MAX_POS_DELTA[1] * velocity[1]

        self._target_pose[0] += dx
        self._target_pose[1] += dy
        self._target_pose[2] += self.BASE_MAX_THETA_DELTA * velocity[2]

        return {'base_pose': self._target_pose}

    def step(self, obs):
        pygame.event.pump()
        start = self.joy.get_button(7)
        back = self.joy.get_button(6)

        if self._episode_ended:
            if start and not self._prev_start:
                self._prev_start = start
                self._prev_back = back
                return 'reset_env'
            self._prev_start = start
            self._prev_back = back
            return None

        if back and not self._prev_back:
            self._episode_ended = True
            self._prev_start = start
            self._prev_back = back
            return 'end_episode'

        self._prev_start = start
        self._prev_back = back
        return self._read_action(obs)

    def save_prompt(self, writer):
        return _gamepad_save_prompt(self.joy, writer)


# Handle SIGTERM
def handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
signal.signal(signal.SIGTERM, handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mobile_base', action='store_true', help='Teleop the mobile base instead of the arm')
    args = parser.parse_args()

    if args.mobile_base:
        teleop = GamepadTeleop()
        try:
            teleop.run()
        finally:
            if teleop.vehicle:
                teleop.vehicle.stop_control()
    else:
        teleop = GamepadArmTeleop()
        teleop.run()
