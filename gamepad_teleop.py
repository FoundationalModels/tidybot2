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
# Base control:
#   Start/Back -> enable/disable, hold LB or RB to move
#   Left stick -> X/Y velocity, Right stick X -> angular velocity
#   LB = local frame, RB = global frame
#
# Arm control:
#   Start -> reset arm (home + open gripper)
#   Hold LB or RB (dead-man's switch) to move arm:
#     Left stick X/Y -> EE X/Y translation
#     D-pad up/down  -> EE +Z/-Z translation
#     Right stick X  -> EE roll rotation
#     Right stick Y  -> EE pitch rotation
#     X button       -> EE +yaw rotation
#     Y button       -> EE -yaw rotation
#   Hold A -> close gripper, hold B -> open gripper (works without dead-man's switch)

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
                    # self.vehicle.set_target_position(self.vehicle.x + 1.5 * target_velocity)

                elif last_enabled:
                    print('Robot disabled')
                    last_enabled = False

            time.sleep(0.01)

class GamepadArmTeleop:
    # Max EE delta per control cycle (at 10 Hz)
    ARM_MAX_POS_DELTA = np.array([0.02, 0.02, 0.02])  # meters
    ARM_MAX_ROT_DELTA = 0.05  # radians
    ARM_GRIPPER_DELTA = 0.1  # gripper position [0=open, 1=closed]

    def __init__(self):
        self.joy = Joystick(0)  # Logitech F710
        self.arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            self.arm_manager.connect()
        except Exception as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
        self.arm = self.arm_manager.Arm()

    def run(self):
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
        print(f'(Gamepad: {self.joy.get_name()}, {self.joy.get_numaxes()} axes, '
              f'{self.joy.get_numbuttons()} buttons, {self.joy.get_numhats()} hats)')

        arm_ready = False
        last_enabled = False
        prev_start = False
        target_pos = None
        target_quat = None
        gripper_pos = 0.0

        while True:
            pygame.event.pump()

            # Reset arm on Start button press (rising edge)
            start = self.joy.get_button(7)
            if start and not prev_start:
                print('Resetting arm...')
                self.arm.reset()
                # Initialize targets from actual state after reset
                state = self.arm.get_state()
                target_pos = state['arm_pos'].copy()
                target_quat = state['arm_quat'].copy()
                gripper_pos = 0.0
                arm_ready = True
                print('Arm ready')
            prev_start = start

            if not arm_ready:
                time.sleep(0.1)
                continue

            # Dead-man's switch: hold LB or RB to enable all control
            left_bumper = self.joy.get_button(4)
            right_bumper = self.joy.get_button(5)
            if left_bumper or right_bumper:
                if not last_enabled:
                    print('Control enabled')
                    last_enabled = True

                # Gripper: hold A to close, hold B to open
                a_button = self.joy.get_button(0)
                b_button = self.joy.get_button(1)
                gripper_pos = float(np.clip(
                    gripper_pos + self.ARM_GRIPPER_DELTA * (float(a_button) - float(b_button)),
                    0.0, 1.0,
                ))

                # Translation: accumulate deltas onto local target (avoids feedback oscillation)
                dx = -self.joy.get_axis(1)   # left stick Y axis -> X (forward/back)
                dy = -self.joy.get_axis(0)   # left stick X axis -> Y (left/right)
                dz = float(self.joy.get_hat(0)[1])  # D-pad up=+1, down=-1
                pos_deltas = apply_deadzone(np.array([dx, dy]))
                target_pos += self.ARM_MAX_POS_DELTA * np.array([pos_deltas[0], pos_deltas[1], dz])

                # Rotation: right stick -> roll/pitch, X/Y buttons -> +/-yaw
                droll  = -self.joy.get_axis(3)             # right stick X
                dpitch = self.joy.get_axis(4)              # right stick Y
                dyaw   = float(self.joy.get_button(2)) - float(self.joy.get_button(3))  # X=+yaw, Y=-yaw
                rot_deltas = apply_deadzone(np.array([droll, dpitch]), deadzone_size=0.1)
                delta_rot = R.from_euler('ZYX', [
                    self.ARM_MAX_ROT_DELTA * dyaw,
                    self.ARM_MAX_ROT_DELTA * rot_deltas[1],
                    self.ARM_MAX_ROT_DELTA * rot_deltas[0],
                ])
                target_quat = (delta_rot * R.from_quat(target_quat)).as_quat()
                if target_quat[3] < 0:
                    np.negative(target_quat, out=target_quat)

                self.arm.execute_action({
                    'arm_pos': target_pos,
                    'arm_quat': target_quat,
                    'gripper_pos': np.array([gripper_pos]),
                })
            else:
                if last_enabled:
                    print('Control disabled')
                    last_enabled = False

            time.sleep(0.1)

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
