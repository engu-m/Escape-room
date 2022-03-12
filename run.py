"""play a game and see it on terminal"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from textwrap import wrap
from tqdm import tqdm
import matplotlib.pyplot as plt

from Environment import EscapeRoomEnvironment
from constants import *
import viz
from string_recorder import StringRecorder


def render_frame(
    env,
    agent_name,
    run_ratio,
    episode_ratio,
    all_actions,
    episode_reward,
    fps,
    terminal,
    display_on_screen,
):
    """render given frame to terminal"""
    frame = env.render(
        # "\n".join(wrap(all_actions, env.grid_w + 2)), # uncomment to add all actions from episode to frame
        f"reward : {episode_reward}",
        f"room : {env.room_ratio}",
        f"run : {run_ratio}",
        f"episode : {episode_ratio}",
        agent_name,
    )
    if display_on_screen:
        os.system("cls")
        sys.stdout.write(frame)
        time.sleep(fps)
        if terminal:
            # see final frame a little longer
            time.sleep(fps)
    return frame


def episodes(agent_name, agent, env, run_ratio, **episode_parameters):
    """run an episode for the agent in one env/room"""
    num_episodes = episode_parameters["num_episodes"]
    episodes_nb_to_show = episode_parameters["episodes_nb_to_show"]
    fps = episode_parameters["fps"]
    n_first_episode_visit = episode_parameters["n_first_episode_visit"]
    n_last_episode_visit = episode_parameters["n_last_episode_visit"]
    save_frames_to_gif = episode_parameters["save_frames_to_gif"]
    display_on_screen = episode_parameters["display_on_screen"]
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
        action = agent.agent_start(
            (*env.start_loc, 0), seed=episode + num_episodes * int(get_nb_from_ratio(run_ratio))
        )
        all_actions = ""
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
                    display_on_screen,
                )
                if save_frames_to_gif:
                    rec.record_frame(frame)

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
                    episode_ratio = f"{episode+1}/{num_episodes}"
                    frame = render_frame(
                        env,
                        agent_name,
                        run_ratio,
                        episode_ratio,
                        all_actions,
                        episode_reward,
                        fps,
                        True,
                        display_on_screen,
                    )
                    if save_frames_to_gif:
                        # save terminal recording to gif
                        video_dir = Path("Escape-Room-RL/video")
                        video_dir.mkdir(exist_ok=True)
                        rec.record_frame(frame)
                        rec.make_gif(
                            str(
                                video_dir
                                / (
                                    f"agent-{agent_name}_"
                                    f"run-{get_nb_from_ratio(run_ratio)}_"
                                    f"room-{get_nb_from_ratio(env.room_ratio)}_"
                                    f"episode-{get_nb_from_ratio(episode_ratio)}"
                                    ".gif"
                                )
                            ),
                        )
                all_episode_rewards.append(episode_reward)
                break
    return all_episode_rewards


def run(run_ratio, agent_name, agent, rooms, episode_params):
    """make agent go through all rooms, one at a time"""
    for room_nb, (room_name, room_params) in enumerate(rooms):
        # init env
        env_params["room_params"] = room_params
        env_params["room_name"] = room_name
        env_params["room_ratio"] = f"{room_nb+1}/{len(rooms)}"
        env = EscapeRoomEnvironment(env_params=env_params)
        # run all episodes
        episodes(agent_name, agent, env, run_ratio, **episode_params)


global rec
rec = StringRecorder()
dict_all_episode_rewards = {}
num_runs = 3
for agent_name, agent in tqdm(agents.items()):
    for run_nb in tqdm(range(num_runs)):
        run_ratio = f"{run_nb+1}/{num_runs}"
        run(run_ratio, agent_name, agent, rooms, episode_params)
