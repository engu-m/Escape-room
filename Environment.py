"""Abstract environment base class for RL-Glue-py.
"""

import gym
import numpy as np


class EscapeRoomEnvironment:
    """Implements the environment for an RLGlue environment
    Note:
        __init__, env_start, env_step, env_cleanup are required
        methods.
    """

    def __init__(self, env_params={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        reward = None
        state = None  # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)

        # for render purpose only
        self.agent_last_action = None

        self.grid_h = env_params.get("grid_height", 5)
        self.grid_w = env_params.get("grid_width", 5)
        self.grid_shape = (self.grid_h, self.grid_w)

        # The start is in the middle of the bottom row of the room
        self.start_loc = (self.grid_h - 1, self.grid_w // 2)
        # The door is in the middle of the top row of the room
        self.goal_loc = (0, self.grid_w // 2)
        # There is an obstacle in front of the door
        self.obstacle_loc = (self.goal_loc[0] + 1, self.goal_loc[1])

        # map bounds
        self.UP_map_bound = [(-1, y) for y in range(-1, self.grid_w + 1)]
        self.DOWN_map_bound = [(self.grid_h, y) for y in range(-1, self.grid_w + 1)]
        self.RIGHT_map_bound = [(x, -1) for x in range(-1, self.grid_h + 1)]
        self.LEFT_map_bound = [(x, self.grid_w) for x in range(-1, self.grid_h + 1)]
        self.forbidden_locs = (
            self.UP_map_bound
            + self.DOWN_map_bound
            + self.RIGHT_map_bound
            + self.LEFT_map_bound
            + [self.obstacle_loc]
        )
        self.forbidden_locs = list(set(self.forbidden_locs))  # render all locs unique
        # The key is in the bottom right corner
        self.key_loc = (self.grid_h - 1, self.grid_w - 1)
        # The player does not have the key in the beginning
        self.got_key = 0

        assert (
            self.key_loc not in self.forbidden_locs
        ), "key location init is forbidden, try another location"
        assert (
            self.start_loc not in self.forbidden_locs
        ), "start location init is forbidden, try another location"
        assert (
            self.goal_loc not in self.forbidden_locs
        ), "goal location init is forbidden, try another location"

    def start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        self.got_key = 0  # take the key from the player
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = (*self.agent_loc, self.got_key)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term

    def render(self):
        """render the current state to terminal"""
        lut = {
            0: " ",
            1: gym.utils.colorize("P", "blue"),  # player
            2: gym.utils.colorize("D", "green"),  # door
            3: gym.utils.colorize("K", "yellow"),  # key
            4: "|",  # wall left/right
            5: "-",  # wall up/down
            6: gym.utils.colorize("X", "red"),  # obstacles
            7: gym.utils.colorize("↑", "blue"),  # player up
            8: gym.utils.colorize("←", "blue"),  # player left
            9: gym.utils.colorize("↓", "blue"),  # player down
            10: gym.utils.colorize("→", "blue"),  # player right
        }

        r = np.zeros(self.grid_shape, dtype="int8")

        r[self.goal_loc] = 2  # door
        if self.got_key == 0:
            r[self.key_loc] = 3  # key
        r[self.obstacle_loc] = 6

        agent_state = self.reward_state_term[1]

        if agent_state is not None:
            agent_loc = agent_state[:2]
            if self.agent_last_action is not None:
                r[agent_loc] = self.agent_last_action + 7  # up -> 7, left -> 8, etc.
            else:
                r[agent_loc] = 1  # default when no movement

        r = np.pad(
            r, 1, mode="constant", constant_values=4
        )  # left/right walls everywhere outside the room
        r[0][:] = 5  # top wall
        r[-1][:] = 5  # bottom wall
        r_str = ""
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r_str += lut[r[i, j]]
            r_str += "\n"
        return r_str

    def step(self, action):

        if action == 0:  # UP
            possible_next_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
        elif action == 1:  # LEFT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] - 1)
        elif action == 2:  # DOWN
            possible_next_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
        elif action == 3:  # RIGHT
            possible_next_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
        else:
            raise Exception(
                str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!"
            )

        reward = -1
        terminal = False

        if possible_next_loc not in self.forbidden_locs:
            self.agent_loc = possible_next_loc
            if self.agent_loc == self.goal_loc and self.got_key == 1:
                reward = 10
                terminal = True
            elif self.agent_loc == self.key_loc and self.got_key == 0:
                self.got_key = 1
                reward = 1

        state = (*self.agent_loc, self.got_key)
        self.reward_state_term = (reward, state, terminal)
        self.agent_last_action = action
        return self.reward_state_term
