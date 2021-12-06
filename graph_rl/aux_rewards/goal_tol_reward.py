from . import AuxReward

class GoalTolReward(AuxReward):
    """Auxiliary reward based on chosen goal_tol."""

    def __init__(self, func):
        self._f = func

    def __call__(self, obs, action, env_reward):
        return self._f(action["goal_tol"])
