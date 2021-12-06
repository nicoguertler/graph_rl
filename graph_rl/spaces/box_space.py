import numpy as np
from gym import spaces as gym_spaces

from .space import Space

class BoxSpace(Space):

    def __init__(self, low, high, dtype = np.float32):
        super().__init__()

        assert np.shape(low) == np.shape(high)

        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self._difference = self.high - self.low
        self.n = np.shape(self.low)[0]

        self.bounded_below = -np.inf < self.low
        self.bounded_above = self.high < np.inf

    def contains(self, value):
        return np.all(np.logical_and(self.low < value, value < self.high))

    def get_flat_space(self, indices = None):
        if indices is None:
            return self
        else:
            subspace = BoxSpace(
                    low = self.low[indices],
                    high = self.high[indices], 
                    dtype = self.dtype
                    )
            return subspace

    def get_flat_dim(self):
        """Get dimension of flat space associated to this space."""
        return self.n

    def flatten_value(self, value):
        return value

    def unflatten_value(self, flat_value):
        """Maps a vector from the space returned by flatten_value back to this space."""
        return flat_value

    def sample(self):
        res = np.empty(self.low.shape)
        for a in [False, True]:
            for b in [False, True]:
                applies = (b == self.bounded_below) & (a == self.bounded_above)
                # bounded
                if b == True and a == True:
                    res[applies] = self._difference[applies]*np.random.random(size = np.sum(applies)) \
                            + self.low[applies]
                # bounded from below
                if b == True and a == False:
                    res[applies] = np.random.exponential(size = np.sum(applies)) + self.low[applies]
                # bounded from above
                if b == False and a == True:
                    res[applies] = -np.random.exponential(size = np.sum(applies)) + self.high[applies]
                # unbounded
                if b == False and a == False:
                    res[applies] = np.random.normal(size = np.sum(applies))
        return res

    def get_gym_space(self):
        return gym_spaces.Box(self.low, self.high, dtype = self.dtype)

    def __eq__(self, other):
        if not isinstance(other, BoxSpace):
            return False
        return np.all(np.equal(self.low, other.low)) \
                and np.all(np.equal(self.high, other.high)) \
                and self.dtype == other.dtype

    @staticmethod
    def concatenate(*args):
        for arg in args[1:]:
            assert arg.dtype == args[0].dtype
        low = np.concatenate([arg.low for arg in args])
        high = np.concatenate([arg.high for arg in args])
        return BoxSpace(low, high, dtype = args[0].dtype)

    @staticmethod
    def get_from_gym_space(gym_space):
        return BoxSpace(low = gym_space.low, high = gym_space.high, 
                dtype = gym_space.dtype)

    def __str__(self):
        return "BoxSpace(low = {}; high = {}; dtype = {})".format(self.low, self.high, self.dtype)
