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
    "num_runs": 5,
    "num_episodes": 50,
    "fps": 0,
    "n_first_episode_visit": n_first_episode_visit,
    "n_last_episode_visit": n_last_episode_visit,
    "save_frames_to_gif": False,
}

num_episodes = run_params["num_episodes"]
episodes_nb_to_show = range(10)  # show 10 first episodes
episodes_nb_to_show = range(num_episodes - 10, num_episodes)  # show 10 last episodes
episodes_nb_to_show = [0, num_episodes - 1]  # show first and last episodes only
episodes_nb_to_show = [
    min(k * num_episodes // 5, num_episodes - 1) for k in range(10 + 1)
]  # show all k*10% episodes
episodes_nb_to_show = [0, num_episodes - 1]  # show first and last episode
episodes_nb_to_show = []  # show no episode on terminal

run_params["episodes_nb_to_show"] = episodes_nb_to_show

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
        "No key no obstacle",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, w // 2), "top-left", "bottom-left"],
            "need_key": False,
        },
    ),
    (
        "No key but obstacles",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, w // 2), "top-left", "bottom-right"],
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
