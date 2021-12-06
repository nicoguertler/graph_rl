from abc import ABC, abstractmethod

class AuxReward:

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, obs, action, env_reward):
        pass

    def update(self, env_info):
        """This is called in every time step."""

        pass
