import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

action_to_emoji = {0: "↑", 1: "←", 2: "↓", 3: "→"}

save_dir = Path("Escape-Room-RL/viz")
save_dir.mkdir(exist_ok=True)


def plot_q_value_estimation(q_value, n_runs, save=True, show=False):
    fig, ax = plt.subplots(2, 4)
    for action in range(4):
        for got_key in range(2):
            ax[got_key, action].imshow(q_value[:, :, got_key, action])
            ax[got_key, action].set_title(f"{action_to_emoji[action]}" + got_key * " - got_key")
            ax[got_key, action].set_xticks([])
            ax[got_key, action].set_yticks([])
    fig.suptitle(f"Q value estimation after {n_runs} runs")
    fig.tight_layout()
    if save:
        plt.savefig(save_dir / f"Q value estimation after {n_runs}")
    if show:
        plt.show()


def plot_n_last_visits(all_state_visits, n_runs, save=True, show=False):
    average_state_visits = all_state_visits.sum(axis=(0, -1))
    average_state_visits = average_state_visits[
        ::-1, :
    ]  # reverse x axis because internal array coordinates =/= screen coordinates

    figsize = average_state_visits.shape[::-1]
    figsize = (figsize[0], figsize[1] + 2)
    fig, ax = plt.subplots(figsize=figsize)
    plt.pcolormesh(average_state_visits, edgecolors="gray", linewidth=2)
    plt.title(f"Visits during\n the last {all_state_visits.shape[0]}\n episodes")
    plt.axis("off")

    cm = plt.get_cmap()
    plt.subplots_adjust(bottom=0.05, top=0.8, right=0.75, left=0.05)
    cax = plt.axes([0.82, 0.05, 0.075, 0.75])
    cbar = plt.colorbar(cax=cax)
    # plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(
            save_dir / f"{all_state_visits.shape[0]} last episode visits out of {n_runs} runs.png"
        )
