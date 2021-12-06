from abc import ABC, abstractmethod

class Space(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_flat_space(self):
        pass

    @abstractmethod
    def flatten_value(self, value):
        pass

    @abstractmethod
    def get_flat_dim(self):
        """Get dimension of flat space associated to this space."""
        pass

    @abstractmethod
    def unflatten_value(self, flat_value):
        """Maps a vector from the space returned by flatten_value back to this space."""
        pass

    @abstractmethod
    def contains(self, value):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @staticmethod
    @abstractmethod
    def concatenate(*args):
        pass

    @staticmethod
    @abstractmethod
    def get_from_gym_space(gym_space):
        pass

    @abstractmethod
    def __str__(self):
        pass
