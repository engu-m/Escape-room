"""play a game and see it on terminal"""

from constants import run_params, env_params, agent_params, agents
from run import complete_run

complete_run(env_params, run_params, agents, agent_params)
