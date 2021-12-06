import numpy as np

from . import AuxReward

class PowerReward(AuxReward):
    """Auxiliary reward proportional to negative power output."""

    def __init__(self, factor, map_to_vel, map_to_force):
        self._c = factor
        self._map_to_vel = map_to_vel
        self._map_to_force = map_to_force

    def __call__(self, obs, action, env_reward):
        vel = self._map_to_vel(obs)
        force = self._map_to_force(obs, action)
        return -self._c*np.dot(vel, force)
