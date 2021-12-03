from abc import ABC, abstractmethod

class Policy(ABC):

    def __init__(self, name, observation_space, action_space):
        super().__init__()
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def __call__(self, observation, algo_info, testing = False):
        pass
