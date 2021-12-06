from . import AuxReward

class EnvReward(AuxReward):
    """Environment reward as auxiliary reward."""

    def __init__(self, weight):
        self._weight = weight

    def __call__(self, obs, action, env_reward):
        return self._weight*env_reward
