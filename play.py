"""play a game and see it on terminal"""

import os
import sys
import time
import numpy as np
import keyboard

from Environment import EscapeRoomEnvironment

init_params = {
    "grid_width": 6,
    "grid_height": 8,
}

env = EscapeRoomEnvironment(env_info=init_params)
env.env_start()

key_to_action = {"haut": 0, "gauche": 1, "bas": 2, "droite": 3}

# iterate
while True:
    os.system("cls")
    sys.stdout.write(env.env_render())
    event = keyboard.read_event()
    if event.event_type == keyboard.KEY_DOWN and event.name in key_to_action.keys():
        action = key_to_action[event.name]
        reward, state, term = env.env_step(action)
        time.sleep(0.1)
        if term:
            # render final frame
            sys.stdout.write(env.env_render())
            break
