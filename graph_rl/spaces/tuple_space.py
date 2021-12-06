import numpy as np

from . import Space

class TupleSpace(Space):

    def __init__(self, *args):
        super().__init__()

        self._component_spaces = args

    def contains(self, value):
        for c, c_space in zip(value, self._component_spaces):
            if not c_space.contains(c):
                return False
        return True

    @property
    def length(self):
        return len(self._component_spaces)

    def get_flat_space(self, indices = None):
        if indices is None:
            indices = range(self.length)

        spaces = [self._component_spaces[i] for i in indices]

        for space in spaces[1:]:
            assert isinstance(space, type(spaces[0]))

        return spaces[0].concatenate(*spaces)

    def flatten_value(self, value):
        flat_value = np.concatenate(
                [space.flatten_value(c) for c, space in zip(value, self._component_spaces)])
        return flat_value

    def sample(self):
        return tuple(space.sample() for space in self._component_spaces)

    def get_gym_space(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not (isinstance(other, TupleSpace) and 
                self.length == other.length):
            return False
        for c1, c2 in zip(self._component_spaces, other._component_spaces):
            if c1 != c1:
                return False
        return True

    @staticmethod
    def concatenate(*args):
        component_spaces = []
        for arg in args:
            component_spaces.extend(arg._component_spaces)
        return TupleSpace(component_spaces)

    @staticmethod
    def get_from_gym_space(gym_space):
        raise NotImplementedError

    def __str__(self):
        res = "TupleSpace("
        for c in self._component_spaces:
            res += str(c)
        res += ")"
        return res
        
