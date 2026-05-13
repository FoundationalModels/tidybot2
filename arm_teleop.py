import time
import numpy as np
from arm_server import ArmManager
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY, POLICY_CONTROL_PERIOD
from policies import TeleopPolicy

manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
manager.connect()
arm = manager.Arm()

policy = TeleopPolicy()  # Starts Flask web server on port 5000 (URL printed above)

try:
    while True:
        print('Resetting arm...')
        arm.reset()
        print('Arm reset. Press "Start episode" in the phone web app.')

        policy.reset()  # Blocks until "Start episode" is pressed
        print('Episode started.')

        start_time = time.time()
        for step_idx in range(10000):
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            obs = arm.get_state()
            obs['base_pose'] = np.zeros(3)

            action = policy.step(obs)

            if action is None:
                continue
            elif isinstance(action, dict):
                arm.execute_action(action)
            elif action == 'end_episode':
                print('Episode ended. Press "Reset env" in the web app to continue.')
            elif action == 'reset_env':
                break
finally:
    arm.close()
