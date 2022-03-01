"""Agents for EscapeRoom"""
import numpy as np

import numpy as np


class BaseAgent:
    def __init__(self, agent_init_info) -> None:
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.tuple_state = agent_init_info["tuple_state"]
        self.num_states_x, self.num_states_y = self.tuple_state
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        # num_states_x (height), num_states_y (width), got_key or not (2), num_actions (4)
        self.q = np.zeros((*self.tuple_state, 2, self.num_actions))

    def agent_start(self, state, seed):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's env_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
        current_q = self.q[state[0], state[1], state[2], :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)  # random action selection
        else:
            action = self.argmax(current_q)  # greedy action selection
        self.prev_state = state
        self.prev_action = action
        self.rand_generator = np.random.RandomState(seed)
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.q[self.prev_state, self.prev_action] = reward

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []
        for i, _ in enumerate(q_values):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)


class QLearningAgent(BaseAgent):
    def __init__(self, agent_init_info) -> None:
        super().__init__(agent_init_info)

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        state_x, state_y, got_key = state
        prev_state_x, prev_state_y, prev_got_key = self.prev_state
        current_q = self.q[state_x, state_y, got_key, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        best_action = self.argmax(current_q)
        self.q[prev_state_x, prev_state_y, prev_got_key, self.prev_action] += self.step_size * (
            reward
            + self.discount * current_q[best_action]
            - self.q[prev_state_x, prev_state_y, prev_got_key, self.prev_action]
        )

        self.prev_state = state
        self.prev_action = action
        return action


class ExpectedSarsa(BaseAgent):
    def __init__(self, agent_init_info) -> None:
        super().__init__(agent_init_info)

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        state_x, state_y, got_key = state
        prev_state_x, prev_state_y, prev_got_key = self.prev_state
        current_q = self.q[state_x, state_y, got_key, :]

        best_action = self.argmax(current_q)
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = best_action

        expectation = self.epsilon * current_q.mean() + (1 - self.epsilon) * current_q[best_action]
        self.q[prev_state_x, prev_state_y, prev_got_key, self.prev_action] += self.step_size * (
            reward
            + self.discount * expectation
            - self.q[prev_state_x, prev_state_y, prev_got_key, self.prev_action]
        )

        self.prev_state = state
        self.prev_action = action
        return action


class TDAgent:
    def __init__(self, agent_info={}):

        # self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Policy will be given, recall that the goal is to accurately estimate its corresponding value function.
        self.policy = agent_info.get("policy")
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount")
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size")

        self.values = np.zeros(self.policy.shape[:-1])  # exlude action dimension

    def agent_start(self, state):

        action = self.rand_generator.choice(range(self.policy.shape[-1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_step(self, reward, state):

        target = reward + self.discount * self.values[state]
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (
            target - self.values[self.last_state]
        )

        action = self.rand_generator.choice(range(self.policy.shape[-1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_end(self, reward):

        target = reward
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (
            target - self.values[self.last_state]
        )
