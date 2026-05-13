# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy

def should_save_episode(writer):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False

    # Prompt user whether to save episode
    while True:
        user_input = input('Save episode (y/n)? ').strip().lower()
        if user_input == 'y':
            return True
        if user_input == 'n':
            print('Discarding episode')
            return False
        print('Invalid response')

def run_episode(env, policy, writer=None, save_prompt_fn=None,
                pre_reset_msg='Press "Start episode" in the web app when ready to start new episode',
                post_episode_msg='Teleop is now active. Press "Reset env" in the web app when ready to proceed.'):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    if pre_reset_msg is not None:
        print(pre_reset_msg)
    policy.reset()
    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        obs = env.get_obs()

        # Get action
        action = policy.step(obs)

        # No action if teleop not enabled
        if action is None:
            continue

        # Execute valid action on robot
        if isinstance(action, dict):
            env.step(action)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(obs, action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')

            prompt = save_prompt_fn if save_prompt_fn is not None else should_save_episode
            if writer is not None and prompt(writer):
                writer.flush_async()
                writer.wait_for_flush()

            print(post_episode_msg)

        # Ready for env reset
        elif action == 'reset_env':
            break

def main(args):
    # Create env
    if args.sim:
        from mujoco_env import MujocoEnv
        if args.teleop or args.gamepad:
            env = MujocoEnv(show_images=True)
        else:
            env = MujocoEnv()
    elif args.both_bots:
        from real_env import RealEnv
        env = RealEnv()
    elif args.gamepad and args.robot == 'base':
        from real_env import BaseOnlyEnv
        env = BaseOnlyEnv()
    else:
        from real_env import ArmOnlyEnv
        env = ArmOnlyEnv()

    # Create policy
    if args.gamepad:
        if args.robot == 'base':
            from gamepad_teleop import GamepadBasePolicy
            policy = GamepadBasePolicy()
        else:
            from gamepad_teleop import GamepadArmPolicy
            policy = GamepadArmPolicy()
        save_prompt_fn = policy.save_prompt
    elif args.teleop:
        policy = TeleopPolicy()
        save_prompt_fn = None
    else:
        policy = RemotePolicy()
        save_prompt_fn = None

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            if args.gamepad:
                run_episode(env, policy, writer, save_prompt_fn=save_prompt_fn,
                            pre_reset_msg=None,
                            post_episode_msg='Press Start on gamepad when ready to proceed.')
            else:
                run_episode(env, policy, writer, save_prompt_fn=save_prompt_fn)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--both_bots', action='store_true')
    parser.add_argument('--gamepad', action='store_true')
    parser.add_argument('--robot', default='arm', choices=['arm', 'base'])
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-dir', default='data/demos')
    args = parser.parse_args()
    if args.both_bots and args.gamepad:
        parser.error('--both_bots and --gamepad cannot both be set')
    main(args)
