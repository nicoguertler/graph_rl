from abc import ABC, abstractmethod

class GlobalAlgorithm(ABC):

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def add_experience(self, transition, sess_info):
        pass

    @abstractmethod
    def learn(self, sess_info):
        pass
