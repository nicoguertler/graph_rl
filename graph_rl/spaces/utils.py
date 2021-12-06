from .space import Space
from .gym_conv_dict import gym_conv_dict

def space_from_gym_space(gym_space):

    if not isinstance(gym_space, Space):
        cls = gym_space.__class__
        return gym_conv_dict[cls].get_from_gym_space(gym_space)
    else:
        return gym_space

