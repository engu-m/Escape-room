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

env_params = {
    "grid_width": 4,
    "grid_height": 5,
}

fps = 0.2


agent_info = {
    "num_actions": 4,
    "epsilon": 0.1,
    "tuple_state": (env_params["grid_height"], env_params["grid_width"]),
    "discount": 1,
    "step_size": 0.8,
    "seed": 3,
}

agents = {
    "ExpectedSarsa": ExpectedSarsa(agent_init_info=agent_info),
    "QLearningAgent": QLearningAgent(agent_init_info=agent_info),
}

num_runs = 500
runs_nb_to_show = range(10)  # show 10 first runs
runs_nb_to_show = range(num_runs - 10, num_runs)  # show 10 last runs
runs_nb_to_show = [
    min(k * num_runs // 10, num_runs - 1) for k in range(10 + 1)
]  # show all k*10% runs
runs_nb_to_show = [0, num_runs - 1]  # show first and last runs only
runs_nb_to_show = []  # show no run on terminal

n_first_run_visit = 120  # number of first run visits to show
n_last_run_visit = 300  # number of last run visits to show


def run(agent_name, agent, env, **run_parameters):
    num_runs = run_parameters["num_runs"]
    runs_nb_to_show = run_parameters["runs_nb_to_show"]
    fps = run_parameters["fps"]
    n_first_run_visit = run_parameters["n_first_run_visit"]
    n_last_run_visit = run_parameters["n_last_run_visit"]
    viz_results = run_parameters["viz_results"]
    viz_params = run_parameters["viz_params"]

    last_state_visits = np.zeros(
        (n_last_run_visit, env_params["grid_height"], env_params["grid_width"], 2)
    )
    first_state_visits = np.zeros(
        (n_first_run_visit, env_params["grid_height"], env_params["grid_width"], 2)
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

    if viz_results:
        save_dir = Path("Escape-Room-RL/viz") / agent_name
        save_dir.mkdir(exist_ok=True)
        viz.plot_best_action_per_state(agent.q, num_runs, save_directory=save_dir, **viz_params)
        viz.plot_n_first_visits(first_state_visits, num_runs, save_directory=save_dir, **viz_params)
        viz.plot_n_last_visits(last_state_visits, num_runs, save_directory=save_dir, **viz_params)
        viz.plot_q_value_estimation(agent.q, num_runs, save_directory=save_dir, **viz_params)


if __name__ == "__main__":
    run_parameters = {
        "num_runs": num_runs,
        "runs_nb_to_show": runs_nb_to_show,
        "fps": fps,
        "n_first_run_visit": n_first_run_visit,
        "n_last_run_visit": n_last_run_visit,
        "viz_results": True,
        "viz_params": {"save": True, "show": True},
    }

    agent_name = "ExpectedSarsa"
    agent = agents["ExpectedSarsa"]
    env = EscapeRoomEnvironment(env_info=env_params)
    run(agent_name, agent, env, **run_parameters)
