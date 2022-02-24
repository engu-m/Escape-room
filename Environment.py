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

    def __init__(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """

        # Note, we can setup the following variables later, in env_start() as it is equivalent.
        # Code is left here to adhere to the note above, but these variables are initialized once more
        # in env_start() [See the env_start() function below.]

        reward = None
        state = None  # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)

        # AN ASIDE: Observation is a general term used in the RL-Glue files that can be interachangeably
        # used with the term "state" for our purposes and for this assignment in particular.
        # A difference arises in the use of the terms when we have what is called Partial Observability where
        # the environment may return states that may not fully represent all the information needed to
        # predict values or make decisions (i.e., the environment is non-Markovian.)

        self.grid_h = env_info.get("grid_height", 5)
        self.grid_w = env_info.get("grid_width", 5)
        self.grid_shape = (self.grid_h, self.grid_w)

        self.start_loc = (self.grid_h - 1, self.grid_w // 2)
        # Goal location is the bottom-right corner. (max x, max y).
        self.goal_loc = (0, self.grid_w // 2)
        # The door is in the middle of the top line of the room
        self.obstacle_loc = (self.goal_loc[0] + 1, self.goal_loc[1])
        # There is an obstacle in front of the door

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
        self.forbidden_locs = list(set(self.forbidden_locs))

        self.key_loc = (self.grid_h - 1, self.grid_w - 1)
        assert (
            self.key_loc not in self.forbidden_locs
        ), "key location init is forbidden, try another location"
        assert (
            self.start_loc not in self.forbidden_locs
        ), "start location init is forbidden, try another location"
        assert (
            self.goal_loc not in self.forbidden_locs
        ), "goal location init is forbidden, try another location"
        # The key is in the bottom right corner
        self.got_key = False
        # The player does not have the key in the beginning

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = (*self.agent_loc, self.got_key)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term[1]

    def env_render(self):
        """render the current state to terminal
        0 : background (' ')
        1 : player ('P')
        2 : door ('D')
        3 : key ('K')
        4 : left/right wall ('|')
        5 : top/bottom wall ('-')
        6 : obstacle ('X')
        """
        lut = {
            0: " ",
            1: gym.utils.colorize("P", "blue"),
            2: gym.utils.colorize("D", "green"),
            3: gym.utils.colorize("K", "yellow"),
            4: "|",
            5: "-",
            6: gym.utils.colorize("X", "red"),
        }

        r = np.zeros(self.grid_shape, dtype="int8")

        r[self.goal_loc] = 2  # door
        if not self.got_key:
            r[self.key_loc] = 3  # key
        r[self.obstacle_loc] = 6

        agent_state = self.reward_state_term[1]

        if agent_state is not None:
            agent_loc = agent_state[:2]
        r[agent_loc] = 1

        r = np.pad(r, 1, mode="constant", constant_values=4)
        r[0][:] = 5
        r[-1][:] = 5
        r_str = ""
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r_str += lut[r[i, j]]
            r_str += "\n"
        return r_str
