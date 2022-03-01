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

init_params = {
    "grid_width": 3,
    "grid_height": 7,
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

plot_q_value = False

if plot_q_value:
    fig, ax = plt.subplots(2, 4)
    for action in range(4):
        for got_key in range(2):
            ax[got_key, action].imshow(agent.q[:, :, got_key, action])
            ax[got_key, action].set_title(
                f"Action {action_to_emoji[action]}" + got_key * ", got_key"
            )
    fig.suptitle("Q value estimation after the last run")
    fig.tight_layout()
    plt.show()

plot_visits = True
if plot_visits:

    average_state_visits = all_state_visits.sum(axis=(0, -1))
    average_state_visits = average_state_visits[
        ::-1, :
    ]  # reverse x axis because internal array coordinates =/= screen coordinates

    figsize = average_state_visits.shape[::-1]
    figsize = (figsize[0], figsize[1] + 2)
    fig, ax = plt.subplots(figsize=figsize)
    plt.pcolormesh(average_state_visits, edgecolors="gray", linewidth=2)
    plt.title(f"Visits during\n the last 10\n episodes")
    plt.axis("off")

    cm = plt.get_cmap()
    plt.subplots_adjust(bottom=0.05, top=0.8, right=0.75, left=0.05)
    cax = plt.axes([0.82, 0.05, 0.075, 0.75])
    cbar = plt.colorbar(cax=cax)
    # plt.tight_layout()
    save_dir = Path("Escape-Room-RL/viz")
    save_dir.mkdir(exist_ok=True)
    plt.show()
    plt.savefig(save_dir / f"{n_last_run_visit} last visits out of {num_runs} runs.png")

print("done")
