"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

from Environment import EscapeRoomEnvironment
from agent import QLearningAgent, ExpectedSarsa
import viz

init_params = {
    "grid_width": 4,
    "grid_height": 5,
}

fps = 0.2
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

num_runs = 5000
runs_nb_to_show = range(10)  # show 10 first runs
runs_nb_to_show = range(num_runs - 10, num_runs)  # show 10 last runs
runs_nb_to_show = [
    min(k * num_runs // 10, num_runs - 1) for k in range(10 + 1)
]  # show all k*10% runs
runs_nb_to_show = [0, num_runs - 1]  # show first and last runs only
runs_nb_to_show = []  # show no run on terminal

n_first_run_visit = 120  # number of first run visits to show
n_last_run_visit = 300  # number of last run visits to show

last_state_visits = np.zeros(
    (n_last_run_visit, init_params["grid_height"], init_params["grid_width"], 2)
)
first_state_visits = np.zeros(
    (n_first_run_visit, init_params["grid_height"], init_params["grid_width"], 2)
)

for run in range(num_runs):
    reward, state, term = env.start()
    action = agent.agent_start((*env.start_loc, 0), seed=run)
    # iterate
    while True:
        if run in runs_nb_to_show:
            os.system("cls")
            sys.stdout.write(env.render())
            time.sleep(fps)

        # step in env and agent
        reward, state, term = env.step(action)
        action = agent.agent_step(reward, state)

        # track visits if necessary
        if run < n_first_run_visit:
            first_state_visits[(run, *state)] += 1
        if run >= num_runs - n_last_run_visit:
            last_state_visits[(run - (num_runs - n_last_run_visit), *state)] += 1

        if term:
            break

plot_params = {"save": True, "show": False}
viz.best_action_per_state(agent.q, num_runs, **plot_params)
viz.plot_n_first_visits(first_state_visits, num_runs, **plot_params)
viz.plot_n_last_visits(last_state_visits, num_runs, **plot_params)
viz.plot_q_value_estimation(agent.q, num_runs, **plot_params)
