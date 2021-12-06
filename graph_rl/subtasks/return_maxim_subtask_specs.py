from abc import ABC, abstractmethod

from .subtask_specs import SubtaskSpec
from ..spaces import space_from_gym_space


class ReturnMaximSubtaskSpec(SubtaskSpec):
    """Specification of return maximization subtask, i.e., obs, reward etc."""

    def __init__(self, max_n_actions=None):
        super().__init__()
        self._max_n_actions = max_n_actions

    @property
    @abstractmethod
    def obs_space(self):
        """Observation space."""

        pass

    @abstractmethod
    def map_to_obs(self, env_obs, parent_info, ep_step):
        """Maps environment observation, parent info and episode step to observation."""

        pass

    @abstractmethod
    def get_reward(self, obs, act, new_obs, parent_info, ep_step, env_info):
        """Get reward of transition."""

        pass

    @abstractmethod
    def step_update(self, env_info, parent_info, sess_info):
        """Update internal state of subtask spec in each time step."""

        pass

    def reset(self):
        """Reset state of subtask sepc."""

        pass

    @property
    def max_n_actions(self):
        """Maximum number of actions before subtask ends."""

        return self._max_n_actions


class EnvRMSubtaskSpec(ReturnMaximSubtaskSpec):
    """Return maximization subtask specification based on environment."""

    def __init__(self, env, max_n_actions=None):
        super().__init__(max_n_actions)
        self._env = env
        self._env_obs_space = space_from_gym_space(env.observation_space)
        self._cum_reward = 0.

    @property
    def obs_space(self):
        """Use environment observation space."""

        return self._env_obs_space

    def map_to_obs(self, env_obs, parent_info, ep_step):
        """Return environment observation."""

        return env_obs

    def get_reward(self, obs, act, new_obs, parent_info, ep_step, env_info):
        """Return cumulative environment reward.
        
        Reward has been accumulated since las call to get_reward 
        or beginning of episode.
        """

        rew = self._cum_reward + env_info.reward
        self._cum_reward = 0

        return rew

    def step_update(self, env_info, parent_info, sess_info):
        """Add env reward of time step to cumulative reward."""

        self._cum_reward += env_info.reward
