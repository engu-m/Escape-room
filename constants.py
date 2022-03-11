"""All constants used in other programs """

from agent import ExpectedSarsa

env_params = {
    "grid_width": 4,
    "grid_height": 4,
}

fps = 0.15


agent_info = {
    "num_actions": 4,
    "epsilon": 0.1,
    "tuple_state": (env_params["grid_height"], env_params["grid_width"]),
    "discount": 1,
    "step_size": 0.8,
    "seed": 3,
}

agents = {
    # "QLearningAgent": QLearningAgent(agent_init_info=agent_info),
    "ExpectedSarsa": ExpectedSarsa(agent_init_info=agent_info),
}

num_episodes = 200
episodes_nb_to_show = range(10)  # show 10 first episodes
episodes_nb_to_show = range(num_episodes - 10, num_episodes)  # show 10 last episodes
episodes_nb_to_show = [0, num_episodes - 1]  # show first and last episodes only
episodes_nb_to_show = []  # show no episode on terminal
episodes_nb_to_show = [
    min(k * num_episodes // 5, num_episodes - 1) for k in range(10 + 1)
]  # show all k*10% episodes

n_first_episode_visit = 120  # number of first episode visits to show
n_last_episode_visit = 300  # number of last episode visits to show


episode_params = {
    "num_episodes": num_episodes,
    "episodes_nb_to_show": episodes_nb_to_show,
    "fps": fps,
    "n_first_episode_visit": n_first_episode_visit,
    "n_last_episode_visit": n_last_episode_visit,
    "save_to_gif": True,
    "viz_results": False,
    "viz_params": {
        "save": False,
        "show": False,
        "cmap": "magma",
        "max_fontsize": 40,
        "block_show": True,
    },
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
            "obstacle_locations": [(1, 4 // 2), "top-left", "bottom-left"],
            "need_key": False,
        },
    ),
    (
        "No key but obstacles",
        {
            "door_location": "top-middle",
            "key_location": None,
            "agent_location": "bottom-middle",
            "obstacle_locations": [(1, 4 // 2), "top-left", "bottom-right"],
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
            "obstacle_locations": [(1, 4 // 2), "top-right", "bottom-left"],
            "need_key": True,
        },
    ),
]
