"""All constants used in other programs """
from itertools import product

from agent import ExpectedSarsa, QLearningAgent

action_to_emoji = {0: "↑", 1: "←", 2: "↓", 3: "→"}

w, h = 4, 4

env_params = {
    "grid_width": w,
    "grid_height": h,
}


agent_params = {
    "num_actions": 4,
    "epsilon": 0.1,
    "grid_shape": (env_params["grid_height"], env_params["grid_width"]),
    "discount": 1,
    "step_size": 0.8,
}

agents = {
    "QLearningAgent": QLearningAgent,
    "ExpectedSarsa": ExpectedSarsa,
}

n_first_episode_visit = 120  # number of first episode visits to show
n_last_episode_visit = 300  # number of last episode visits to show

run_params = {
    "num_runs": 3,
    "num_episodes": 30,
    "fps": 0.15,
    "n_first_episode_visit": n_first_episode_visit,
    "n_last_episode_visit": n_last_episode_visit,
    "save_frames_to_gif": True,
}

num_episodes = run_params["num_episodes"]


## episodes to show
episodes_to_show = range(10)  # show 10 first episodes
episodes_to_show = range(num_episodes - 10, num_episodes)  # show 10 last episodes
episodes_to_show = [0, num_episodes - 1]  # show first and last episodes only
episodes_to_show = [
    min(k * num_episodes // 5, num_episodes - 1) for k in range(10 + 1)
]  # show all k*10% episodes
episodes_to_show = [0, num_episodes - 1]  # show first and last episode
episodes_to_show = [num_episodes - 1]  # show only the last one
episodes_to_show = []  # show no episode on terminal
episodes_to_show = product(
    [1], [0], [6], list(episodes_to_show)
)  # second agent, first run, 7th room

## episodes to save
episodes_to_save = [num_episodes - 1]  # save only the last one
episodes_to_save = product(
    [0, 1], [0], [6], list(episodes_to_save)
)  # second agent, first run, 7th room

run_params["episodes_to_show"] = list(episodes_to_show)
run_params["episodes_to_save"] = list(episodes_to_save)

viz_params = {
    "save": False,
    "show": False,
    "cmap": "magma",
    "max_fontsize": 40,
    "block_show": True,
}


rooms = [
    (
        "No key no obstacle close to the door",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "top-right",
            "obstacle_locations": [],
            "need_key": False,
        },
    ),
    (
        "No key corridor",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": list(
                product(range(h), list(range(w // 2)) + list(range(w // 2 + 1, w)))
            ),
            "need_key": False,
        },
    ),
    (
        "No key no obstacle far from the door",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [],
            "need_key": False,
        },
    ),
    (
        "No key but obstacle in front of the door",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, w // 2), "bottom-left"],
            "need_key": False,
        },
    ),
    (
        "No key many obstacles",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, w // 2), "top-left", "bottom-right", "top-right"],
            "need_key": False,
        },
    ),
    (
        "Key bottom-right but no obstacles",
        {
            "door_location": "top-middle",
            "key_location": "bottom-right",
            "agent_location": "bottom-middle",
            "obstacle_locations": [],
            "need_key": True,
        },
    ),
    (
        "Key bottom right + obstacles",
        {
            "door_location": "top-left",
            "key_location": "bottom-right",
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, w // 2), "top-right", "bottom-left"],
            "need_key": True,
        },
    ),
]


env_params["rooms"] = rooms


def get_nb_from_ratio(s: str):
    """If s is like 15/2508, this returns only 15, the first number padded
    by the length of 2508 so 0015
    This allows the files to be alphabetically sorted"""
    nb, total = s.split("/")
    fmt_str = f"{{:0{len(total)}d}}"
    nb_padded = fmt_str.format(int(nb))
    return nb_padded
