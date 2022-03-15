"""play a game and see it on terminal"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from Environment import EscapeRoomEnvironment
from constants import get_nb_from_ratio
from string_recorder import StringRecorder


def render_frame(
    term,
    env,
    agent_name,
    run_ratio,
    episode_ratio,
    episode_reward,
    fps,
    save,
    show,
    rec,
):
    """create string to represent frame.
    If required, add it to string recorder
    If terminal, save it as gif"""
    # create string
    frame = env.render(
        agent_name,
        f"run : {run_ratio}",
        f"room : {env.room_ratio}",
        f"episode : {episode_ratio}",
        f"reward : {episode_reward}",
    )
    if show:
        os.system("cls")
        sys.stdout.write(frame)
        time.sleep(fps)
        if term:
            time.sleep(2 * fps)
    if save:
        rec.record_frame(frame)
        if term:
            # make it last a little more
            rec.record_frame(frame)
            rec.record_frame(frame)
            rec.record_frame(frame)
            # save terminal recording to video
            video_dir = Path("./Escape-Room-RL/video")
            video_dir.mkdir(exist_ok=True)
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


def complete_run(env_params, run_params, agents, agent_params):
    """complete run of all runs + rooms + episodes"""
    # run params
    num_runs = run_params["num_runs"]
    num_episodes = run_params["num_episodes"]
    fps = run_params["fps"]
    episodes_to_show = run_params["episodes_to_show"]
    episodes_to_save = run_params["episodes_to_save"]
    # env params
    rooms = env_params["rooms"]
    num_rooms = len(rooms)
    # agent_params
    num_actions = agent_params["num_actions"]
    grid_shape = agent_params["grid_shape"]
    tuple_state = (*grid_shape, 2)
    rec = StringRecorder()

    state_actions = np.zeros(
        (len(agents), num_runs, num_rooms, num_episodes, *tuple_state, num_actions)
    )
    rewards_continuous = np.zeros((len(agents), num_runs, num_episodes * num_rooms))
    rewards_room_by_room = np.zeros((len(agents), num_runs, num_rooms, num_episodes))
    value_last_episode = np.zeros((len(agents), num_runs, num_rooms, *tuple_state, num_actions))

    for agent_nb, (agent_name, agent_class) in enumerate(tqdm(agents.items())):
        for run_nb in tqdm(range(num_runs)):
            agent_params["seed"] = run_nb
            agent = agent_class(agent_params)
            run_ratio = f"{run_nb+1}/{num_runs}"
            run_total_episode = 0
            for room_nb, (room_name, room_params) in enumerate(rooms):
                # initialize env
                env_params["room_params"] = room_params
                env_params["room_name"] = room_name
                env_params["room_ratio"] = f"{room_nb+1}/{len(rooms)}"
                env = EscapeRoomEnvironment(env_params=env_params)

                iteration = 0
                for episode in tqdm(range(num_episodes), leave=False):
                    # start agent
                    reward, state, term = env.start()
                    action = agent.agent_start(state, seed=episode + num_episodes * run_nb)
                    episode_reward = reward

                    iteration += 1
                    # iterate
                    while True:
                        # record episode
                        save = (agent_nb, run_nb, room_nb, episode) in episodes_to_save
                        show = (agent_nb, run_nb, room_nb, episode) in episodes_to_show
                        if save or show:
                            episode_ratio = f"{episode+1}/{num_episodes}"
                            render_frame(
                                False,
                                env,
                                agent_name,
                                run_ratio,
                                episode_ratio,
                                episode_reward,
                                fps,
                                save,
                                show,
                                rec,
                            )

                        # step in env and agent
                        reward, state, term = env.step(action)
                        action = agent.agent_step(reward, state)

                        # store statistics
                        state_actions[(agent_nb, run_nb, room_nb, episode, *state, action)] += 1
                        episode_reward += agent_params["discount"] ** iteration * reward

                        iteration += 1

                        if term:
                            # store statistics
                            rewards_continuous[agent_nb, run_nb, run_total_episode] = episode_reward
                            rewards_room_by_room[
                                agent_nb, run_nb, room_nb, episode
                            ] = episode_reward
                            run_total_episode += 1

                            # record episode
                            save = (agent_nb, run_nb, room_nb, episode) in episodes_to_save
                            show = (agent_nb, run_nb, room_nb, episode) in episodes_to_show
                            if save or show:
                                episode_ratio = f"{episode+1}/{num_episodes}"
                                render_frame(
                                    True,
                                    env,
                                    agent_name,
                                    run_ratio,
                                    episode_ratio,
                                    episode_reward,
                                    fps,
                                    save,
                                    show,
                                    rec,
                                )
                            break
                value_last_episode[agent_nb, run_nb, room_nb] = agent.q
    state_visits = state_actions.sum(axis=-1)

    return {
        "state_actions": state_actions,
        "state_visits": state_visits,
        "rewards_continuous": rewards_continuous,
        "rewards_room_by_room": rewards_room_by_room,
        "value_last_episode": value_last_episode,
    }
