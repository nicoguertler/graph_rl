import numpy as np
from gym import spaces as gym_spaces

from .space import Space

class MultiBinarySpace(Space):
    def __init__(self, n, dtype = np.int8):
        super().__init__()
        self.n = n
        self.dtype = dtype

    def contains(self, value):
        return np.all(np.logical_or(value == 0, value == 1))

    def get_flat_space(self):
        return self

    def flatten_value(self, value):
        return value

    def sample(self):
        res = np.random.randint(0, 2, size = self.n, dtype = np.int8)
        return np.array(res, dtype = self.dtype)

    def get_gym_space(self):
        return gym_spaces.MultiBinary(n, dtype = self.dtype)

    def __eq__(self, other):
        if not isinstance(other, MultiBinarySpace):
            return False
        return self.n == self.other and self.dtype == other.dtype

    @staticmethod
    def concatenate(*args):
        for arg in args[1:]:
            assert arg.dtype == args[0].dtype
        return MultiBinarySpace(sum([arg.n for arg in args]), dtype = args[0].dtype)

    @staticmethod
    def get_from_gym_space(gym_space):
        return MultiBinarySpace(gym_space.n, dtype = gym_space.dtype)

    def __str__(self):
        return "MultiBinarySpace: n = {}; dtype = {}".format(self.n, self.dtype)




