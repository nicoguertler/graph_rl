from abc import ABC, abstractmethod

class AuxReward:

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, obs, action):
        pass
