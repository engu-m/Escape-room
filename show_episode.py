"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Environment import EscapeRoomEnvironment
from agent import QLearningAgent, ExpectedSarsa

init_params = {
    "grid_width": 4,
    "grid_height": 5,
}

env = EscapeRoomEnvironment(env_info=init_params)

key_to_action = {"haut": 0, "gauche": 1, "bas": 2, "droite": 3}
action_to_key = {0: "haut", 1: "gauche", 2: "bas", 3: "droite"}
action_to_emoji = {0: "↑", 1: "←", 2: "↓", 3: "→"}


agent_info = {
    "num_actions": 4,
    "epsilon": 0.1,
    "tuple_state": (init_params["grid_height"], init_params["grid_width"]),
    "discount": 1,
    "step_size": 0.8,
    "seed": 3,
}

# agent = QLearningAgent(agent_init_info=agent_info)
agent = ExpectedSarsa(agent_init_info=agent_info)
num_runs = 10000
show_n_last_runs = 10

for run in tqdm(range(num_runs)):
    reward, state, term = env.start()
    action = agent.agent_start((*env.start_loc, 0), seed=run)
    # iterate
    while True:
        if run > num_runs - show_n_last_runs:
            # Render the game
            os.system("cls")
            sys.stdout.write(env.render())
            time.sleep(0.1)

        reward, state, term = env.step(action)
        action = agent.agent_step(reward, state)

        if term:
            break

fig, ax = plt.subplots(2, 4)
for action in range(4):
    for got_key in range(2):
        ax[got_key, action].imshow(agent.q[:, :, got_key, action])
        ax[got_key, action].set_title(f"Action {action_to_emoji[action]}" + got_key * ", got_key")

# fig.tight_layout()
plt.show()

print("done")
