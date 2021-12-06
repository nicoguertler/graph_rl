import numpy as np

from . import AuxReward

class ConstantReward(AuxReward):
    """Constant auxiliary reward."""

    def __init__(self, constant_reward):
        self._reward = constant_reward

    def __call__(self, obs, action, env_reward):
        return self._reward
