from pickle import FALSE
import numpy as np
from time import time
from Environment import EscapeRoomEnvironment

def value_iteration_policy(env):
    h,w = env.grid_h,env.grid_w
    policy = np.zeros((h,w,2,4))
    values = np.zeros((h,w,2))
    stable = False
    i = 0
    while stable == False:
        new_values = np.zeros_like(values)
        for x in range(h):
            for y in range(w):
                for got_key in range(2):
                    next_state_values = np.zeros(4)
                    rewards = np.zeros(4)                   
                    if (x,y) != env.goal_loc or got_key == 0:
                        for action in range(4):
                            env.agent_loc = (x,y)
                            env.got_key = got_key
                            reward, next_state, _ = env.step(action)
                            rewards[action] = reward
                            next_state_values[action] = values[next_state]
                        best_action = np.argmax(rewards + next_state_values)
                        new_value = np.max(rewards + next_state_values)
                        policy[x][y][got_key] = best_action
                        new_values[x][y][got_key] = new_value
        if np.all(new_values == values):
            stable = True
        else :
            i+=1
        if i%10000 == 0:
            print(i)
            print(values)
        values = new_values
    return policy,values

init_params = {
    "grid_width": 4,
    "grid_height": 4,
}

env = EscapeRoomEnvironment(env_info=init_params)

policy,values = value_iteration_policy(env)
env.start()
print(policy)
print(values)
