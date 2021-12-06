import numpy as np
from gym import spaces as gym_spaces

from .space import Space

class BinarySpace(Space):
    def __init__(self):
        super().__init__()
        self.dtype = bool

    def contains(self, value):
        if not isinstace(value, np.bool_) or isinstance(value, bool):
            return False
        else:
            return True

    def get_flat_space(self):
        return self

    def flatten_value(self, value):
        return np.array([value])

    def get_flat_dim(self):
        """Get dimension of flat space associated to this space."""
        return 1

    def unflatten_value(self, flat_value):
        """Maps a vector from the space returned by flatten_value back to this space."""
        return flat_value[0]

    def sample(self):
        res = np.random.randint(0, 2, dtype = np.int8)
        return bool(res)

    def get_gym_space(self):
        return gym_spaces.MultiBinary(1, dtype = self.int8)

    def __eq__(self, other):
        return isinstance(other, BinarySpace)

    @staticmethod
    def concatenate(*args):
        for arg in args:
            assert isinstance(arg, BinarySpace)
        return MultiBinarySpace(len(args), dtype = args[0].dtype)

    @staticmethod
    def get_from_gym_space(gym_space):
        return BinarySpace()

    def __str__(self):
        return "BinarySpace"
