import numpy as np
import gym


class DummyEnv(gym.GoalEnv):  
    """Env that only holds observation_space and action_space.

    Dummy environment can be used to communicate observation and action space 
    to stable baselines algorithms."""

    metadata = {'render.modes': ['human']}   

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass
