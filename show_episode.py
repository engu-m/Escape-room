"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from Environment import EscapeRoomEnvironment
from agent import QLearningAgent, ExpectedSarsa
import viz

init_params = {
    "grid_width": 4,
    "grid_height": 5,
}

env = EscapeRoomEnvironment(env_info=init_params)

key_to_action = {"haut": 0, "gauche": 1, "bas": 2, "droite": 3}
action_to_key = {0: "haut", 1: "gauche", 2: "bas", 3: "droite"}


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
show_n_last_runs = 0
n_last_run_visit = 10  # number of last run to show visit

all_state_visits = np.zeros(
    (n_last_run_visit, init_params["grid_height"], init_params["grid_width"], 2)
)
for run in tqdm(range(num_runs)):
    reward, state, term = env.start()
    action = agent.agent_start((*env.start_loc, 0), seed=run)
    # iterate
    while True:
        # if run % int(num_runs / 10) == 0 and run > 20: # show the k*n/10 th runs
        if run >= num_runs - show_n_last_runs:  # show the n last runs
            # Render the game
            os.system("cls")
            sys.stdout.write(env.render())
            time.sleep(0.2)

        reward, state, term = env.step(action)
        action = agent.agent_step(reward, state)
        if run >= num_runs - n_last_run_visit:
            all_state_visits[(run - (num_runs - n_last_run_visit), *state)] += 1

        if term:
            break

plot_params = {"save": True, "show": True}
viz.best_action_per_state(agent.q, num_runs, **plot_params)
viz.plot_q_value_estimation(agent.q, num_runs, **plot_params)
viz.plot_n_last_visits(all_state_visits, num_runs, **plot_params)
