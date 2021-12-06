import numpy as np
from gym import spaces as gym_spaces

from . import Space

class DictSpace(Space):

    def __init__(self, space_dict):
        super().__init__()

        self._space_dict = space_dict

        # calculate dimension of flat space associated to this space
        self._flat_dim = self._get_flat_dim_uncached()

    def contains(self, value):
        if not isinstance(value, dict):
            return False
        for key, c in value.items():
            if not key in self._space_dict or not self._space_dict[key].contains(c):
                return False
        return True

    @property
    def length(self):
        return len(self._space_dict)

    def keys(self):
        return self._space_dict.keys()

    def get_flat_space(self, keys = None):
        if keys is None:
            keys = self._space_dict.keys()
        # NOTE: Need sorting in order to avoid different orders of keys.
        keys = sorted(keys)

        spaces = [self._space_dict[key].get_flat_space() for key in keys]

        return spaces[0].concatenate(*spaces)

    def _get_flat_dim_uncached(self, keys = None):
        if keys is None:
            keys = self._space_dict.keys()
        keys = sorted(keys)

        subspace_dims = [self._space_dict[key].get_flat_dim() for key in keys]

        return np.sum(subspace_dims)

    def get_flat_dim(self, keys = None):
        """Get dimension of flat space associated to this space."""
        # used cached value int this case. NOTE: This assumes the space is not modified after creation!
        if keys is None:
            return self._flat_dim
        else:
            return self._get_flat_dim_uncached(keys)

    def flatten_value(self, value, keys = None):
        if keys is None:
            keys = self._space_dict.keys()
        keys = sorted(keys)

        flat_sub_values = [self._space_dict[key].flatten_value(value[key]) for key in keys]
        flat_value = np.concatenate(flat_sub_values)
        return flat_value

    def unflatten_value(self, flat_value, keys = None):
        """Maps a vector from the space returned by flatten_value back to this space.
        
        Assumes that flat_value contains only contributions from the subspaces with keys 
        listed in keys. Assumes that space (and subspaces) have not been modified since 
        creation."""
        if keys is None:
            keys = self._space_dict.keys()
        keys = sorted(keys)

        # dimensions of subspaces
        dims = [self._space_dict[key].get_flat_dim() for key in keys]
        indices = [0]
        for dim in dims:
            indices.append(indices[-1] + dim)
        value = {key: self._space_dict[key].unflatten_value(flat_value[indices[i]:indices[i + 1]]) for i, key in enumerate(keys)}
        return value
        
    def sample(self):
        return {key: space.sample() for key, space in self._space_dict.items()}

    def get_gym_space(self):
        dict_of_gym_spaces = {key: space.get_gym_space() for key, space in self._space_dict.items()}
        return gym_spaces.Dict(dict_of_gym_spaces)

    def __eq__(self, other):
        if not (isinstance(other, DictSpace) and 
                self.keys() == other.keys()):
            return False
        for c1, c2 in zip(self._space_dict.values(), other._space_dict.values()):
            if c1 != c1:
                return False
        return True

    @staticmethod
    def concatenate(*args):
        space_dict = args[0]._space_dict
        for arg in args[1:]:
            space_dict.update(arg._space_dict)
        return DictSpace(component_spaces)

    @staticmethod
    def get_from_gym_space(gym_space):
        from .utils import space_from_gym_space
        assert isinstance(gym_space, gym_spaces.Dict)
        return DictSpace({key: space_from_gym_space(gs) for key, gs in gym_space.spaces.items()})

    def __str__(self):
        subspaces_string = ""
        for key, s in self._space_dict.items():
            if subspaces_string != "":
                subspaces_string += ", "
            subspaces_string += "{}: {}".format(key, s)
        return "DictSpace({{ {} }})".format(subspaces_string)

    def __getitem__(self, key):
        return self._space_dict[key]

    def get_subspace(self, keys):
        return DictSpace({key: space for key, space in self._space_dict.items() if key in keys})

