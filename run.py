"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from Environment import EscapeRoomEnvironment
from constants import *
import viz


def render_frame(
    env, agent_name, run_ratio, episode_ratio, all_actions, episode_reward, fps, terminal
):
    """render given frame to terminal"""
    os.system("cls")
    frame = env.render(
        # f"{all_actions}",
        f"reward : {episode_reward}",
        f"room : {env.room_nb} - {env.room_name}",
        f"run : {run_ratio}",
        f"episode : {episode_ratio}",
        agent_name,
    )
    sys.stdout.write(frame)
    time.sleep(fps)
    if terminal:
        time.sleep(4 * fps)
    return frame


def episode(agent_name, agent, env, run_ratio, **episode_parameters):
    """run an episode for the agent in one env/room"""
    num_episodes = episode_parameters["num_episodes"]
    episodes_nb_to_show = episode_parameters["episodes_nb_to_show"]
    fps = episode_parameters["fps"]
    n_first_episode_visit = episode_parameters["n_first_episode_visit"]
    n_last_episode_visit = episode_parameters["n_last_episode_visit"]
    viz_results = episode_parameters["viz_results"]
    viz_params = episode_parameters["viz_params"]

    last_state_visits = np.zeros(
        (n_last_episode_visit, env_params["grid_height"], env_params["grid_width"], 2)
    )
    first_state_visits = np.zeros(
        (n_first_episode_visit, env_params["grid_height"], env_params["grid_width"], 2)
    )
    all_episode_rewards = []
    for episode in tqdm(range(num_episodes)):
        reward, state, term = env.start()
        action = agent.agent_start((*env.start_loc, 0), seed=episode)
        all_actions = ""
        all_frames = []
        episode_reward = reward
        # iterate
        while True:
            if episode in episodes_nb_to_show:
                # render epsiode to terminal with various debug information
                all_actions += viz.action_to_emoji[action]
                episode_ratio = f"{episode}/{num_episodes}"
                frame = render_frame(
                    env,
                    agent_name,
                    run_ratio,
                    episode_ratio,
                    all_actions,
                    episode_reward,
                    fps,
                    False,
                )
                all_frames.append(frame)

            # step in env and agent
            reward, state, term = env.step(action)
            action = agent.agent_step(reward, state)
            episode_reward += reward

            # track visits if necessary
            if episode < n_first_episode_visit:
                first_state_visits[(episode, *state)] += 1
            if episode >= num_episodes - n_last_episode_visit:
                last_state_visits[(episode - (num_episodes - n_last_episode_visit), *state)] += 1

            if term:
                if episode in episodes_nb_to_show:
                    all_actions += viz.action_to_emoji[action]
                    # see final frame a little longer
                    episode_ratio = f"{episode}/{num_episodes}"
                    frame = render_frame(
                        env,
                        agent_name,
                        run_ratio,
                        episode_ratio,
                        all_actions,
                        episode_reward,
                        fps,
                        True,
                    )
                    all_frames.append(frame)
                # all_frames
                all_episode_rewards.append(episode_reward)
                break

    if viz_results:
        save_dir = Path("Escape-Room-RL/viz") / agent_name
        save_dir.mkdir(exist_ok=True, parents=True)
        viz.plot_one_agent_reward(
            all_episode_rewards, agent_name, save_directory=save_dir, **viz_params
        )
        viz.plot_best_action_per_state(agent.q, num_episodes, save_directory=save_dir, **viz_params)
        viz.plot_n_first_visits(
            first_state_visits, num_episodes, save_directory=save_dir, **viz_params
        )
        viz.plot_n_last_visits(
            last_state_visits, num_episodes, save_directory=save_dir, **viz_params
        )
        viz.plot_q_value_estimation(agent.q, num_episodes, save_directory=save_dir, **viz_params)

    return all_episode_rewards


def run(run_ratio, agent_name, agent, rooms, episode_params):
    """make agent go through all rooms, one at a time"""
    for room_nb, (room_name, room_params) in enumerate(rooms):
        # init env
        env_params["room_params"] = room_params
        env_params["room_name"] = room_name
        env_params["room_nb"] = f"{room_nb+1}/{len(rooms)}"
        env = EscapeRoomEnvironment(env_params=env_params)
        # run all episodes
        all_episode_rewards = episode(agent_name, agent, env, run_ratio, **episode_params)
        dict_all_episode_rewards[agent_name] = all_episode_rewards


dict_all_episode_rewards = {}
num_runs = 3
for agent_name, agent in agents.items():
    for run_ratio in tqdm(range(num_runs)):
        run_ratio = f"{run_ratio+1}/{num_runs}"
        run(run_ratio, agent_name, agent, rooms, episode_params)

save_dir = Path("Escape-Room-RL/viz")
save_dir.mkdir(exist_ok=True, parents=True)


# plt.close("all")
# if episode_params["viz_results"]:
#     viz.plot_mutliple_agents_reward(dict_all_episode_rewards, save_directory=save_dir, **episode_params)
