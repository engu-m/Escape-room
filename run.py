"""play a game and see it on terminal"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

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
    save_frames_to_gif,
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
    if fps > 0:
        os.system("cls")
        sys.stdout.write(frame)
        time.sleep(fps)
    if save_frames_to_gif:
        rec.record_frame(frame)
        if term:
            # save terminal recording to gif
            video_dir = Path("Escape-Room-RL/video")
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
    num_runs = run_params["num_runs"]
    rooms = env_params["rooms"]
    rec = StringRecorder()
    for agent_name, agent_class in tqdm(agents.items()):
        for run_nb in tqdm(range(num_runs)):
            agent_params["seed"] = run_nb
            agent = agent_class(agent_params)
            run_ratio = f"{run_nb+1}/{num_runs}"
            for room_nb, (room_name, room_params) in enumerate(rooms):
                # init env
                env_params["room_params"] = room_params
                env_params["room_name"] = room_name
                env_params["room_ratio"] = f"{room_nb+1}/{len(rooms)}"
                env = EscapeRoomEnvironment(env_params=env_params)
                # run all episodes
                num_episodes = run_params["num_episodes"]
                episodes_nb_to_show = run_params["episodes_nb_to_show"]
                fps = run_params["fps"]
                save_frames_to_gif = run_params["save_frames_to_gif"]

                for episode in tqdm(range(num_episodes), leave=False):
                    # start agent
                    action = agent.agent_start(
                        (*env.start_loc, 0), seed=episode + num_episodes * run_nb
                    )
                    reward, state, term = env.start()
                    episode_reward = reward
                    # iterate
                    while True:
                        if episode in episodes_nb_to_show:
                            # render epsiode to terminal with various debug information
                            episode_ratio = f"{episode+1}/{num_episodes}"
                            render_frame(
                                False,
                                env,
                                agent_name,
                                run_ratio,
                                episode_ratio,
                                episode_reward,
                                fps,
                                save_frames_to_gif,
                                rec,
                            )

                        # step in env and agent
                        reward, state, term = env.step(action)
                        action = agent.agent_step(reward, state)
                        episode_reward += reward

                        if term:
                            if episode in episodes_nb_to_show:
                                episode_ratio = f"{episode+1}/{num_episodes}"
                                render_frame(
                                    True,
                                    env,
                                    agent_name,
                                    run_ratio,
                                    episode_ratio,
                                    episode_reward,
                                    fps,
                                    save_frames_to_gif,
                                    rec,
                                )
                            break
