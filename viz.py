"""module for visualizing results of runs/episodes"""

from unicodedata import normalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from pathlib import Path

action_to_emoji = {0: "↑", 1: "←", 2: "↓", 3: "→"}
action_to_text = {0: "haut", 1: "gauche", 2: "bas", 3: "droite"}
action_to_horizontal_alignment = {0: "center", 1: "right", 2: "center", 3: "left"}
action_to_vertical_alignment = {0: "bottom", 1: "center", 2: "top", 3: "center"}

cmap = "magma"
max_fontsize = 45

save_dir = Path("Escape-Room-RL/viz")
save_dir.mkdir(exist_ok=True)


def plot_q_value_estimation(q_value, n_runs, save=True, show=False):
    fig, ax = plt.subplots(2, 4)
    for action in range(4):
        for got_key in range(2):
            ax[got_key, action].imshow(q_value[:, :, got_key, action], cmap=cmap)
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
    # process best value arrays
    best_action_value = np.max(q_value, axis=-1)
    worst_action_value = np.min(q_value, axis=-1)
    scaled_q_value = q_value - worst_action_value[:, :, :, None]
    normalization_vector = np.clip(scaled_q_value.sum(axis=-1), a_min=1e-4, a_max=None)
    all_action_weight = (
        scaled_q_value / normalization_vector[:, :, :, None]
    )  # scale values to be only positive numbers and then compute weight

    # plot both state value map (key or not key)
    for got_key in range(2):
        # plot heatmap
        heatmap_to_plot = best_action_value[:, :, got_key]
        ax[got_key].imshow(heatmap_to_plot, cmap=cmap)

        # title and ticks
        ax[got_key].set_title(got_key * "got_key")
        ax[got_key].set_xticks([])
        ax[got_key].set_yticks([])

        for x, all_action_x_weight in enumerate(all_action_weight[:, :, got_key]):
            for y, weight_action in enumerate(all_action_x_weight):
                for action, weight in enumerate(weight_action):
                    txt = ax[got_key].text(
                        y,
                        x,
                        action_to_emoji[action],
                        ha=action_to_horizontal_alignment[action],
                        va=action_to_vertical_alignment[action],
                        color="white",
                        fontsize=max_fontsize * (weight),  # 0.25 is the minimum
                    )  # this means that perfectly random choices will have an
                    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="black")])

    fig.suptitle(f"Q value best actions after {n_runs} runs")
    fig.tight_layout()
    if save:
        plt.savefig(save_dir / f"Q value best actions after {n_runs} runs")
    if show:
        plt.show()


def plot_n_extreme_visits(all_state_visits, n_runs, save=True, show=False, first_or_last="last"):
    average_state_visits = all_state_visits.sum(axis=(0, -1))
    average_state_visits = average_state_visits[
        ::-1, :
    ]  # reverse x axis because internal array coordinates =/= screen coordinates

    figsize = average_state_visits.shape[::-1]
    figsize = (figsize[0], figsize[1] + 2)
    fig, ax = plt.subplots(figsize=figsize)
    plt.pcolormesh(average_state_visits, edgecolors="gray", linewidth=2, cmap=cmap)
    plt.title(
        f"Visits during\n the {first_or_last} {all_state_visits.shape[0]}\n episodes ({n_runs} runs)"
    )
    plt.axis("off")

    cm = plt.get_cmap()
    plt.subplots_adjust(bottom=0.05, top=0.8, right=0.75, left=0.05)
    cax = plt.axes([0.82, 0.05, 0.075, 0.75])
    cbar = plt.colorbar(cax=cax)
    # plt.tight_layout()
    if save:
        plt.savefig(
            save_dir
            / f"{all_state_visits.shape[0]} {first_or_last} episode visits out of {n_runs} runs.png"
        )
    if show:
        plt.show()


def plot_n_first_visits(all_state_visits, n_runs, save=True, show=False):
    plot_n_extreme_visits(all_state_visits, n_runs, save, show, "first")


def plot_n_last_visits(all_state_visits, n_runs, save=True, show=False):
    plot_n_extreme_visits(all_state_visits, n_runs, save, show, "last")
