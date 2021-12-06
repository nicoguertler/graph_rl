from abc import ABC, abstractmethod

import numpy as np

from ..spaces import BoxSpace, DictSpace, space_from_gym_space
from ..utils import listify, get_obs_space_from_env, get_obs_from_gym
from .subtask_specs import SubtaskSpec

# TODO: Refactor inheritance. Also compare to TimedGoalSubtaskSpecs.

class ShortestPathSubtaskSpec(SubtaskSpec):
    """Specification of shortest path subtask, i.e. criterion of goal achievement etc."""

    def __init__(self, constant_failure_return = False):
        super().__init__()
        self.constant_failure_return = constant_failure_return 

    @abstractmethod
    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        """Maps achieved_goal, desired_goal and parent_info to True/False depending on whether desired_goal was achieved.
        
        Note that parent_info can contain information about the tolerance for achieving a subgoal."""
        pass

    @abstractmethod
    def get_desired_goal(self, env_obs, parent_info):
        """Maps env_obs and parent_info to desired goal in goal_space."""
        pass

    @abstractmethod
    def map_to_goal(self, partial_obs):
        """Maps partial_obs to goal (assumed to be a deterministic map with image in goal_space)."""
        pass

    @abstractmethod
    def map_to_partial_obs(self, env_obs, parent_info, ep_step):
            """Maps env_obs and parent_info to partial observation (assumed to be a deterministic map with image in partial_obs_space)."""
            pass

    @property
    @abstractmethod
    def partial_obs_space(self):
        """Box space for partial observations (partial observation does not contain desired goal)."""
        pass

    @property
    @abstractmethod
    def goal_space(self):
        """Box space containing goals of the subtask."""
        pass

    @property
    @abstractmethod
    def parent_action_space(self):
        """Space which contains actions the parent node is taking. 
        
        The action of a parent determines the the goal of the subtask via the method get_desired_goal."""
        pass

    @property
    @abstractmethod
    def max_n_actions(self):
        """Maximum number of actions before subtask ends."""
        pass

class BoxSPSubtaskSpec(ShortestPathSubtaskSpec):
    """Default subtask spec with goal space being equal to state box space.

    The goal is equal to the parent action, partial observation is partial env 
    observation and achievement criterion uses L2 norm.
    
    If factorization is not None, the space is split up and the L2 norm is 
    applied to each subspace individually (with different thresholds). This 
    can be useful if the goal contains different types of values, e.g., 
    positions and derivatives."""
    def __init__(self, max_n_actions, goal_achievement_threshold, env, factorization = None, 
            constant_failure_return = False):
        super().__init__(constant_failure_return)
        self._max_n_actions = max_n_actions
        # Observation and goal spaces are all identical in this case
        self._space = get_obs_space_from_env(env)
        if factorization is None:
            self.factorization = [range(self._space.get_flat_space().n)]
        else:
            self.factorization = factorization
        self._goal_achievement_threshold = listify(goal_achievement_threshold, 1)
        assert len(self.factorization) == len(self._goal_achievement_threshold)

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for factor, threshold in zip(self.factorization, self._goal_achievement_threshold):
            if np.linalg.norm(achieved_goal[factor] - desired_goal[factor]) >= threshold:
                return False
        return True

    def get_desired_goal(self, env_obs, parent_info):
        # desired goal is equal to parent action, no processing required
        return parent_info.action

    def map_to_goal(self, partial_obs):
        # achieved goal is identical to partial observation
        return partial_obs

    def map_to_partial_obs(self, env_obs, parent_info, ep_step):
        # all levels see full partial observation (but not env goal)
        return get_obs_from_gym(env_obs)

    @property
    def partial_obs_space(self):
        return self._space

    @property
    def goal_space(self):
        return self._space

    @property
    def parent_action_space(self):
        return self._space

    @property
    def max_n_actions(self):
        return self._max_n_actions

class BoxInfoHidingSPSubtaskSpec(BoxSPSubtaskSpec):
    """Subtask spec with goal space composed of only some of the dimensions of the
    state space.

    The goal is equal to the parent action, the partial observation is obtained 
    by picking a subset of the components of the env partial observation.  
    The achieved goal is obtained from the partial obs by concatenating the 
    dimensions in goal_indices. See DefaultSPSubtaskSpec for more details."""
    def __init__(self, max_n_actions, goal_achievement_threshold, partial_obs_indices, 
            goal_indices, env, factorization = None, constant_failure_return = False):
        super(BoxSPSubtaskSpec, self).__init__(constant_failure_return)
        if factorization is None:
            factorization = [range(len(goal_indices))]
        super(BoxInfoHidingSPSubtaskSpec, self).__init__(max_n_actions, 
                goal_achievement_threshold, env, factorization)
        self._partial_obs_indices = partial_obs_indices
        self._goal_indices = range(len(partial_obs_indices)) if goal_indices is None else goal_indices

        self._space = self._space.get_flat_space()
        self._space = self._space.get_flat_space(indices = partial_obs_indices)
        self._goal_space = self._space.get_flat_space(indices = goal_indices)

    def map_to_goal(self, partial_obs):
        # achieved goal is constructed from some of the dimensions of partial obs
        return partial_obs[self._goal_indices]

    def map_to_partial_obs(self, env_obs, parent_info, ep_step):
        # partial observation is constructed by concatenating subset of 
        # components of partial env obs
        return get_obs_from_gym(env_obs)[self._partial_obs_indices]

    @property
    def goal_space(self):
        return self._goal_space

    @property
    def parent_action_space(self):
        return self._goal_space

class DictSPSubtaskSpec(BoxSPSubtaskSpec):
    """Subtask spec with goal space being equal to state dict space.

    The goal is equal to the parent action, partial observation is partial env 
    observation and achievement criterion uses L2 norm.

    The goal space is assumed to be a dict space and the L2 norm in the subspaces 
    is used together with the values in the goal_achievement_threshold dict as 
    thresholds.  This can be useful if the goal contains different types of values, 
    e.g., positions and derivatives."""
    
    def __init__(self, max_n_actions, goal_achievement_threshold, env, 
            constant_failure_return = False):
        super(BoxSPSubtaskSpec, self).__init__(constant_failure_return)
        assert isinstance(goal_achievement_threshold, dict)
        self._max_n_actions = max_n_actions
        # Partial observation, parent action and goal spaces are all identical in this case
        self._space = get_obs_space_from_env(env)
        self._goal_achievement_threshold = goal_achievement_threshold

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for key, threshold in self._goal_achievement_threshold.items():
            if np.linalg.norm(achieved_goal[key] - desired_goal[key]) >= threshold:
                return False
        return True


class DictInfoHidingSPSubtaskSpec(DictSPSubtaskSpec):
    """Subtask spec with partial observation and goal dict space composed of 
    only some of the items of the env observation dict space.

    The desired goal is equal to the parent action. The partial observation is 
    obtained by picking a subset of the keys of the env partial observation. 
    The achieved goal is obtained from the partial obs by concatenating the 
    items with keys in goal_keys. See DefaultSPSubtaskSpec for more details."""
    def __init__(self, max_n_actions, goal_achievement_threshold, partial_obs_keys, 
            goal_keys, env, constant_failure_return = False, max_ep_steps = 1.):
        super(DictInfoHidingSPSubtaskSpec, self).__init__(max_n_actions, 
                goal_achievement_threshold, env, constant_failure_return)
        self._partial_obs_keys = partial_obs_keys
        self._goal_keys = goal_keys

        self._space = self._space.get_subspace(partial_obs_keys)
        if "__ep_time__" in self._partial_obs_keys:
            # add time item to dict space
            new_dict_space = {
                **self._space._space_dict, 
                "__ep_time__": BoxSpace([0.], [1.])
            }
            self._space = DictSpace(new_dict_space)
        self._goal_space = self._space.get_subspace(goal_keys)
        self._max_ep_steps = max_ep_steps

    def map_to_goal(self, partial_obs):
        return {key: value for key, value in partial_obs.items() if key in self._goal_keys}

    def map_to_partial_obs(self, env_obs, parent_info, ep_step):
        p_obs = {key: value for key, value in get_obs_from_gym(env_obs).items() if key in self._partial_obs_keys}
        if "__ep_time__" in self._partial_obs_keys:
            p_obs["__ep_time__"] = np.clip([ep_step/self._max_ep_steps], 0., 1.)
        return p_obs

    @property
    def goal_space(self):
        return self._goal_space

    @property
    def parent_action_space(self):
        return self._goal_space

# subtask specification based on goal based environment
class EnvSPSubtaskSpec(ShortestPathSubtaskSpec):
    def __init__(self, max_n_actions, env, map_to_env_goal, 
            constant_failure_return=False, achievement_reward=0.):
        super().__init__(constant_failure_return)
        self._max_n_actions = max_n_actions
        self._partial_env_obs_space = space_from_gym_space(env.observation_space["observation"])
        self._env_goal_space = space_from_gym_space(env.observation_space["desired_goal"])
        self._map_to_env_goal = map_to_env_goal
        self._env = env
        self._achievement_reward = achievement_reward

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        reward = self._env.compute_reward(achieved_goal, desired_goal, {})
        return reward == self._achievement_reward

    def get_desired_goal(self, env_obs, parent_info):
        return env_obs["desired_goal"]

    def map_to_goal(self, partial_obs):
        return self._map_to_env_goal(partial_obs)

    def map_to_partial_obs(self, env_obs, parent_info, ep_step):
        # highest level sees environment
        return env_obs["observation"]

    @property
    def partial_obs_space(self):
        return self._partial_env_obs_space

    @property
    def goal_space(self):
        return self._env_goal_space

    @property
    def parent_action_space(self):
        return None

    @property
    def max_n_actions(self):
        return self._max_n_actions
