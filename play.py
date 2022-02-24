"""play a game and see it on terminal"""

import os
import sys
from Environment import EscapeRoomEnvironment

init_params = {
    "grid_width": 6,
    "grid_height": 8,
}

env = EscapeRoomEnvironment(env_info=init_params)
env.env_start()

# Render the game
os.system("clear")
sys.stdout.write(env.env_render())
