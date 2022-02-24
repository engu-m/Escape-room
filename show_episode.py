"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np

from Environment import EscapeRoomEnvironment
from TDagent import TDAgent

init_params = {
    "grid_width": 6,
    "grid_height": 8,
}

env = EscapeRoomEnvironment(env_info=init_params)
env.start()

agent_info = {
    "policy": 1 / 4 * np.ones((init_params["grid_height"], init_params["grid_width"], 2, 4)),
    "discount": 1,
    "step_size": 0.8,
}

# agent = TDAgent(agent_info=agent_info)


# iterate
while True:
    # Render the game
    action = np.random.randint(0, 4)
    reward, state, term = env.step(action)
    os.system("cls")
    sys.stdout.write(env.render())

    time.sleep(0.1)
    if term:
        break
