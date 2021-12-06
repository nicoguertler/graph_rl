from gym import spaces as gym_spaces

from . import BoxSpace, TupleSpace, MultiBinarySpace, DictSpace

gym_conv_dict = {
        gym_spaces.Box: BoxSpace,
        gym_spaces.Dict: DictSpace,
        gym_spaces.MultiBinary: MultiBinarySpace
        }
