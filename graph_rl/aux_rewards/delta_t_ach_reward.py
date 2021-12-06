import numpy as np

from . import AuxReward

class DeltaTAchReward(AuxReward):
    """Auxiliary reward proportional to minus the time until achievement.
    
    The time until achievement (delta_t_ach) refers to the quantity 
    used in the HiTS algorithm. This auxiliary reward can therefore 
    be applied to a parent node of a HiTS node. 
    Note that the time until achievement is normalized to the inverval, 
    [0, 1]."""

    def __init__(self, weight):
        """Weight refers to the weight of this reward in the overall reward."""

        self._w = weight

    def __call__(self, obs, action, env_reward):
        delta_t_ach_norm = 0.5*(action["delta_t_ach"][0] + 1.0)
        return -self._w*delta_t_ach_norm
