from . import AuxReward

class EnvReward(AuxReward):
    """Cumulative environment reward as auxiliary reward."""

    def __init__(self, weight):
        self._weight = weight
        self._cum_rew = 0.

    def __call__(self, obs, action, env_reward):
        aux_rew = self._weight*self._cum_rew
        self._cum_rew = 0
        return aux_rew

    def update(self, env_info):
        """Adds env reward to cumulative environment reward every time step."""

        self._cum_rew += env_info.reward
