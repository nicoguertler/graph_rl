from abc import ABC, abstractmethod

import numpy as np

from ..spaces import BoxSpace, DictSpace
from ..utils import listify, get_obs_space_from_env, get_obs_from_gym
from .subtask_specs import SubtaskSpec

class TimedGoal:
    def __init__(self, goal, delta_t_ach, delta_t_comm = None):
        self.goal = goal
        self.delta_t_ach = delta_t_ach
        self.delta_t_comm = delta_t_comm

    @staticmethod
    def from_dict(d):
        return TimedGoal(d["desired_goal"], d["delta_t_ach"][0], 
                d["delta_t_comm"][0] if "delta_t_comm" in d else d["delta_t_ach"][0])

class TimedGoalSubtaskSpec(SubtaskSpec):
    """Specification of timed goal subtask, i.e. criterion of goal achievement etc."""

    def __init__(self, goal_space, partial_obs_space, delta_t_ach = None,  
            delta_t_comm = None, delta_t_max = None, delta_t_min = 0.):
        super().__init__()
        self._goal_space = goal_space
        self._partial_obs_space = partial_obs_space
        self._delta_t_ach = delta_t_ach
        self._delta_t_comm = delta_t_comm
        self._delta_t_max = delta_t_max
        self._delta_t_min = delta_t_min
        # parent action space contains goal, desired time until achievement and (#TODO) commitment time
        space_dict = {"goal": self.goal_space}
        if delta_t_ach is None:
            # normalize time to achievement to [-1, 1] and take into account
            # delta_t_min when adding it to action space
            a_min = self.convert_time(delta_t_min)
            space_dict["delta_t_ach"] = BoxSpace(low = [a_min], high = [1.])
        self._parent_action_space = DictSpace(space_dict)

    @abstractmethod
    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        """Maps achieved_goal, desired_goal and parent_info to True/False depending on whether desired_goal was achieved.
        
        Note that parent_info can contain information about the tolerance for achieving a subgoal."""
        pass

    def unconvert_time(self, t):
        """Map from [-1, 1] interval to time steps."""

        return (t + 1.)/2.*self._delta_t_max

    def convert_time(self, t):
        """Map time to [-1, 1] interval."""

        return t/self._delta_t_max*2. - 1.

    def get_desired_timed_goal(self, env_obs, parent_info):
        """Maps env_obs and parent_info to desired timed goal."""
        # desired goal is equal to parent action, can overwrite for preprocessing
        tg = TimedGoal(
                goal = parent_info.action["goal"],
                delta_t_ach = parent_info.action["delta_t_ach"][0] \
                        if self._delta_t_ach is None else self._delta_t_ach,
                delta_t_comm = parent_info.action["delta_t_ach"][0] \
                        if self._delta_t_comm is None else self._delta_t_comm)
        return tg

    def get_achieved_timed_goal_dict(self, achieved_goal, delta_t_ach, parent_info):
        """delta_t_ach assumed not to be mapped to [-1., 1.] yet"""
        achieved_timed_goal = {
                "goal": achieved_goal,
                "delta_t_ach": [self.convert_time(delta_t_ach)]
                }
        return achieved_timed_goal

    @abstractmethod
    def map_to_goal(self, partial_obs):
        """Maps partial_obs to goal (assumed to be a deterministic map with image in goal_space)."""
        pass

    @abstractmethod
    def map_to_partial_obs(self, env_obs, parent_info):
            """Maps env_obs and parent_info to partial observation (assumed to be a deterministic map with image in partial_obs_space)."""
            pass

    @property
    def partial_obs_space(self):
        """Box space for partial observations (partial observation does not contain desired goal)."""
        return self._partial_obs_space

    @property
    def goal_space(self):
        """Box space containing goals of the subtask."""
        return self._goal_space

    @property
    def parent_action_space(self):
        """Space which contains actions the parent node is taking. 
        
        The action of a parent determines the the timed goal of the subtask via the method get_desired_timed_goal."""
        return self._parent_action_space

class BoxTGSubtaskSpec(TimedGoalSubtaskSpec):
    def __init__(self, goal_achievement_threshold, env, delta_t_max,  delta_t_min = 0., 
            delta_t_ach = None,  delta_t_comm = None, factorization = None):
        # Observation and goal spaces are all identical in this case
        self._space = get_obs_space_from_env(env)
        if factorization is None:
            self.factorization = [range(self._space.get_flat_dim())]
        else:
            self.factorization = factorization
        self._goal_achievement_threshold = listify(goal_achievement_threshold, 1)
        assert len(self.factorization) == len(self._goal_achievement_threshold)
        super(BoxTGSubtaskSpec, self).__init__(self._space, self._space, delta_t_ach, 
                delta_t_comm, delta_t_max = delta_t_max, delta_t_min = delta_t_min)

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for factor, threshold in zip(self.factorization, self._goal_achievement_threshold):
            if np.linalg.norm(achieved_goal[factor] - desired_goal[factor]) >= threshold:
                return False
        return True

    def map_to_goal(self, partial_obs):
        # achieved goal is identical to partial observation
        return partial_obs

    def map_to_partial_obs(self, env_obs, parent_info):
        # all levels see full partial observation (but not env goal)
        return get_obs_from_gym(env_obs)

class BoxInfoHidingTGSubtaskSpec(TimedGoalSubtaskSpec):
    def __init__(self, partial_obs_indices, goal_indices, 
            goal_achievement_threshold, env, delta_t_max, delta_t_min = 0., 
            delta_t_ach = None, delta_t_comm = None, factorization = None):
        self._space = get_obs_space_from_env(env)
        if factorization is None:
            self.factorization = [range(self._space.get_flat_dim())]
        else:
            self.factorization = factorization
        self._goal_achievement_threshold = listify(goal_achievement_threshold, 1)
        assert len(self.factorization) == len(self._goal_achievement_threshold)
        self._partial_obs_indices = partial_obs_indices
        self._goal_indices = goal_indices

        self._space = self._space.get_flat_space(indices=partial_obs_indices)
        self._goal_space = self._space.get_flat_space(indices=goal_indices)
        super().__init__(self._goal_space, self._space, delta_t_ach, delta_t_comm, 
                delta_t_max=delta_t_max, delta_t_min=delta_t_min)

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for factor, threshold in zip(self.factorization, self._goal_achievement_threshold):
            if np.linalg.norm(achieved_goal[factor] - desired_goal[factor]) >= threshold:
                return False
        return True

    def map_to_goal(self, partial_obs):
        # achieved goal is constructed from some of the dimensions of partial obs
        return partial_obs[self._goal_indices]

    def map_to_partial_obs(self, env_obs, parent_info):
        # partial observation is constructed by concatenating subset of 
        # components of partial env obs
        return get_obs_from_gym(env_obs)[self._partial_obs_indices]


"""class DictTGSubtaskSpec(TimedGoalSubtaskSpec):
    def __init__(self, goal_achievement_threshold, env, delta_t_ach = None,  delta_t_comm = None):
        assert isinstance(goal_achievement_threshold, dict)
        # Partial observation, parent action and goal spaces are all identical in this case
        self._space = DictSpace.get_from_gym_space(env.observation_space["observation"])
        self._goal_achievement_threshold = goal_achievement_threshold
        super(BoxTGSubtaskSpec, self).__init__(goal_space = self._space, delta_t_ach, delta_t_comm)

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for key, threshold in self._goal_achievement_threshold.items():
            if np.linalg.norm(achieved_goal[key] - desired_goal[key]) >= threshold:
                return False
        return True"""

class DictInfoHidingTGSubtaskSpecBase(TimedGoalSubtaskSpec):
    """Subtask spec base with partial observation and goal dict space composed of 
    only some of the items of the env observation dict space.

    The partial observation is obtained by picking a subset of the keys of the env 
    partial observation. The achieved goal is obtained from the partial obs by 
    filtering the items with keys in goal_keys. See DefaultSPSubtaskSpec for 
    more details."""
    def __init__(self, partial_obs_keys, goal_keys, env, delta_t_ach = None, 
            delta_t_comm = None, delta_t_max = None, delta_t_min = 0.):
        self._partial_obs_keys = partial_obs_keys
        self._goal_keys = goal_keys
        orig_pos = get_obs_space_from_env(env)

        partial_obs_space = orig_pos.get_subspace(partial_obs_keys)
        goal_space = orig_pos.get_subspace(goal_keys)

        super(DictInfoHidingTGSubtaskSpecBase, self).__init__(goal_space, partial_obs_space, 
                delta_t_ach, delta_t_comm, delta_t_max, delta_t_min)

    def map_to_goal(self, partial_obs):
        return {key: value for key, value in partial_obs.items() if key in self._goal_keys}

    def map_to_partial_obs(self, env_obs, parent_info):
        return {key: value for key, value in get_obs_from_gym(env_obs).items() if key in self._partial_obs_keys}


class DictInfoHidingTGSubtaskSpec(DictInfoHidingTGSubtaskSpecBase):
    """Subtask spec with partial observation and goal dict space composed of 
    only some of the items of the env observation dict space.

    The desired goal is equal to the value corresponding to the goal key in 
    the parent action dict. See DictInfoHidingTGSubtaskSpecBase for more 
    details."""
    def __init__(self, goal_achievement_threshold, partial_obs_keys, 
            goal_keys, env, delta_t_ach=None,  delta_t_comm=None, 
            delta_t_max=None, delta_t_min=0., norms=None):
        assert isinstance(goal_achievement_threshold, dict)
        self._goal_achievement_threshold = goal_achievement_threshold
        if norms is None:
            self._norms = {key: None for key in goal_keys}
        else:
            self._norms = norms

        super(DictInfoHidingTGSubtaskSpec, self).__init__(partial_obs_keys, 
                goal_keys, env, delta_t_ach, delta_t_comm, delta_t_max, 
                delta_t_min)

    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for key, threshold in self._goal_achievement_threshold.items():
            if (np.linalg.norm(achieved_goal[key] - desired_goal[key],ord=self._norms[key])
                    >= threshold):
                return False
        return True

class DictInfoHidingTolTGSubtaskSpec(DictInfoHidingTGSubtaskSpecBase):
    """Subtask spec with partial observation and goal dict space composed of 
    only some of the items of the env observation dict space. Parent action 
    determines not only goal itself but also tolerance for achievement.

    The desired goal is equal to the value corresponding to the goal key in 
    the parent action dict. See DictInfoHidingTGSubtaskSpecBase for more 
    details."""
    def __init__(self, partial_obs_keys, goal_keys, env, max_goal_thresholds = None, 
            delta_t_ach = None, delta_t_comm = None, delta_t_max = None, delta_t_min = 0.):
        super(DictInfoHidingTolTGSubtaskSpec, self).__init__(partial_obs_keys, 
                goal_keys, env, delta_t_ach, delta_t_comm, delta_t_max, delta_t_min)


        # add goal tolerance to parent action space so as higher level can 
        # choose goal achievement tolerance
        space_dict = self._parent_action_space._space_dict
        if max_goal_thresholds is None:
            max_goal_thresholds = {key: 1. for key in goal_keys}
        else:
            assert max_goal_thresholds.keys() == goal_keys
        self._goal_tol_space = DictSpace({key: BoxSpace(low = [0.], high = [value]) 
                for key, value in max_goal_thresholds.items()})
        space_dict["goal_tol"] = self._goal_tol_space 
        self._parent_action_space = DictSpace(space_dict)
    
    def goal_achievement_criterion(self, achieved_goal, desired_goal, parent_info):
        # if the criterion in one of the subspaces is violated, the overall criterion is not met
        for key, threshold in parent_info.action["goal_tol"].items():
            if np.linalg.norm(achieved_goal[key] - desired_goal[key]) >= threshold:
                return False
        return True

    def get_achieved_timed_goal_dict(self, achieved_goal, delta_t_ach, parent_info):
        """delta_t_ach assumed not to be mapped to [-1., 1.] yet"""
        atg = super(DictInfoHidingTolTGSubtaskSpec, self).get_achieved_timed_goal_dict(achieved_goal, delta_t_ach, 
                parent_info)
        atg["goal_tol"] = parent_info.action["goal_tol"]
        return atg
