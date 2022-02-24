"""Agents for EscapeRoom"""
import numpy as np

class TDAgent():

    def __init__(self, agent_info={}):

        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Policy will be given, recall that the goal is to accurately estimate its corresponding value function. 
        self.policy = agent_info.get("policy")
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount")
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size")

        self.values = np.zeros((self.policy.shape[0],))
          
    def agent_start(self, state):

        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action
            
    def agent_step(self, reward, state):

        target = reward + self.discount * self.values[state]
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])

        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_end(self, reward):

        target = reward
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])
