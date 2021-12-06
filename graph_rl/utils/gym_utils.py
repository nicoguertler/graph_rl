from gym import GoalEnv
from gym import spaces

from ..spaces import space_from_gym_space

def get_obs_space_from_env(env):
    if (isinstance(env.observation_space, spaces.Dict) 
        and "observation" in env.observation_space.spaces):
        return space_from_gym_space(env.observation_space["observation"])
    else:
        return space_from_gym_space(env.observation_space)

def get_obs_from_gym(gym_obs):
    if isinstance(gym_obs, dict) and "observation" in gym_obs:
        return gym_obs["observation"]
    else:
        return gym_obs
