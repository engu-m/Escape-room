import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

action_to_emoji = {0: "↑", 1: "←", 2: "↓", 3: "→"}
action_to_text = {0: "haut", 1: "gauche", 2: "bas", 3: "droite"}

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


def best_action_per_state(q_value, n_runs, save=True, show=False):
    fig, ax = plt.subplots(1, 2)
    q_value_best_action_value = np.apply_along_axis(max, -1, q_value)
    q_value_worst_action_value = np.apply_along_axis(min, -1, q_value)
    q_value_best_action_weight = (q_value_best_action_value - q_value_worst_action_value) / (
        1e-4 + (q_value - q_value_worst_action_value[:, :, :, None]).sum(axis=-1)
    )  # scale values to be only positive numbers and then compute weight
    q_value_best_action_int = np.apply_along_axis(np.argmax, -1, q_value)
    q_value_best_action_emoji = np.vectorize(action_to_emoji.get)(q_value_best_action_int)
    # q_value_best_action_text = np.vectorize(action_to_text.get)(q_value_best_action_int)
    for got_key in range(2):
        # plot heatmap
        heatmap_to_plot = q_value_best_action_value[:, :, got_key]
        ax[got_key].imshow(heatmap_to_plot)
        # title and ticks
        ax[got_key].set_title(got_key * "got_key")
        ax[got_key].set_xticks([])
        ax[got_key].set_yticks([])
        # add text on each tile
        for x, best_actions_x_emoji in enumerate(q_value_best_action_emoji[:, :, got_key]):
            for y, emoji in enumerate(best_actions_x_emoji):
                ax[got_key].text(
                    y,
                    x,
                    emoji,
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=90
                    * (q_value_best_action_weight[x, y, got_key] - 0.245),  # 0.25 is the minimum
                )

    fig.suptitle(f"Q value best actions after {n_runs} runs")
    fig.tight_layout()
    if save:
        plt.savefig(save_dir / f"Q value best actions after {n_runs} runs")
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
