from gym import Wrapper

from ..spaces import space_from_gym_space


class FlattenWrapper(Wrapper):
    """Flattens observation and action space."""

    def __init__(self, env):
        super().__init__(env)
        self._action_space = space_from_gym_space(env.action_space)
        self._observation_space = space_from_gym_space(env.observation_space)

        self.action_space = self._action_space.get_flat_space().get_gym_space()
        self.observation_space = self._observation_space.get_flat_space().get_gym_space()

    def step(self, action):
        orig_action = self._action_space.unflatten_value(action)
        orig_obs, reward, done, info = self.env.step(orig_action)
        obs = self._observation_space.flatten_value(orig_obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        orig_obs = self.env.reset(**kwargs)
        obs = self._observation_space.flatten_value(orig_obs)
        return obs



