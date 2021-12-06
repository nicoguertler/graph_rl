from copy import deepcopy
from math import floor

import numpy as np
import torch

from ..data import FlatTransition
from . import OffPolicyAlgorithm
from ..subtasks.timed_goal_subtask_specs import TimedGoal

class HiTS(OffPolicyAlgorithm):
    """Hierarchical reinforcement learning with Timed Subgoals."""

    # sampling strategy "episode" does not make sense for timed goals
    supported_goal_sampling_strategies = {"future", "final"}

    def __init__(self, name, flat_algo_name, flat_algo_kwargs, model, 
            child_failure_penalty, check_status, convert_time, goal_sampling_strategy = "future", 
            buffer_size = 100000, batch_size = 128, n_hindsight_goals = 4, testing_fraction = 0.3, 
            fully_random_fraction = 0.2, bootstrap_testing_transitions = True, 
            use_normal_trans_for_testing = False, use_testing_transitions = True, 
            learn_from_deterministic_episodes = True, log_q_values = True, learning_starts = 0, 
            grad_steps_per_env_step=1, delta_t_min=0.):
            """
            Args:
                check_status (callable): Maps achieved_goal, 
                    desired_timed_goal, step and parent_info to 
                    * achieved: indicates whether timed goal has been achieved
                    * reward: reward upon arriving in achieved_goal
                    * ach_time_up: whether desired time until achievement ran out
                    * comm_time_up: whether commitment time ran out
                delta_t_min: Minimum delta_t. Important for sampling hindsight
                    timed subgoals with HER.
                """
            super(HiTS, self).__init__(name, flat_algo_name, model, fully_random_fraction, 
                    flat_algo_kwargs, buffer_size, batch_size, learning_starts,
                    grad_steps_per_env_step)

            assert goal_sampling_strategy in self.supported_goal_sampling_strategies, \
                    "Goal sampling strategy {} not supported.".format(goal_sampling_strategy)


            self._child_failure_penalty = child_failure_penalty
            self._goal_sampling_strategy = goal_sampling_strategy
            self._check_status = check_status
            self._convert_time = convert_time
            self._n_hindsight_goals = n_hindsight_goals
            self._testing_fraction = testing_fraction
            self._bootstrap_testing_transitions = bootstrap_testing_transitions

            self._use_normal_trans_for_testing = use_normal_trans_for_testing
            self._use_testing_transitions = use_testing_transitions
            self._learn_from_deterministic_episodes = learn_from_deterministic_episodes
            self._log_q_values = log_q_values
            self._delta_t_min = delta_t_min

            self._verbose = False

    def _sample_achieved_timed_goals(self, current_index, episode_start):
        n_transitions = len(self._episode_transitions)
        if self._goal_sampling_strategy == "future":
            low = current_index
            # Adapt number of goals to current_index in a way which is 
            # equivalent to sampling indices uniformly from whole episode
            # and rejecting those which lie in the past. Factor 2 to have
            # number of actual hindsight goals approx. equal to _n_hindsight_goals
            # in expectation.
            n_goals = 2.*(n_transitions - low)/n_transitions*self._n_hindsight_goals
            frac = n_goals - floor(n_goals)
            n_goals_int = floor(n_goals) + (1 if np.random.rand() <= frac else 0)
            indices = np.random.randint(low=low, high=n_transitions, size=n_goals_int)
        elif self._goal_sampling_strategy == "final":
            indices = [-1]

        goals = []
        # time of current new_obs, i.e., time after transition (in env steps)
        current_t = self._episode_transitions[current_index].subtask_tr.info["t"]
        for i in indices:
            # sample delta_t_ach uniformly from interval which would have run out in the 
            # step corresponding to hindsight goal
            rand_shift = np.random.rand()
            goals.append({
                "desired_goal": self._episode_transitions[i].subtask_tr.info["achieved_generalized_goal"]["goal"], 
                "delta_t_ach_obs": [self._convert_time(self._episode_transitions[i].subtask_tr.info["t"] - (current_t - 1) - rand_shift)], 
                "delta_t_ach_new_obs": [self._convert_time(self._episode_transitions[i].subtask_tr.info["t"] - current_t - rand_shift)]
                })
        return goals

    def _add_experience_to_flat_algo(self, parent_info, deterministic_episode, node_is_sink, sess_info):
        if self._learn_from_deterministic_episodes or not deterministic_episode:

            # add transitions of this episode to replay buffer of flat RL algorithm
            # (by applying hindsight goal and action manipulations)
            for trans_index, tr in enumerate(self._episode_transitions):

                # unaltered flat transition
                f_trans_0 = FlatTransition(
                        obs = tr.subtask_tr.obs, 
                        action = tr.subtask_tr.action,
                        reward = tr.subtask_tr.reward,
                        new_obs = tr.subtask_tr.new_obs,
                        done = tr.subtask_tr.info["ach_time_up"])

                # if the node is a sink, do not attempt to manipulate action
                # in hindsight and add original transition to replay buffer
                if node_is_sink:
                    f_trans_base = f_trans_0
                    self._add_to_flat_replay_buffer(f_trans_0)

                # if the node is not a sink consider testing transitions
                # and hindsight action transitions
                else:
                    # did the child node use a deterministic version of its policy?
                    # (testing transition)
                    testing_transition = tr.algo_info["child_be_deterministic"]

                    # boolean indicating whether child achieved subgoal
                    did_child_achieve_subgoal = tr.child_feedback["has_achieved"]

                    # add testing transitions (if enabled)
                    # Optionally, also transitions generated by stochastic 
                    # lower level are used as testing transitions.
                    # Only add transition with penalty if child node failed 
                    # to achieve its subgoal, otherwise do not add any transition.
                    if self._use_testing_transitions \
                       and (testing_transition or self._use_normal_trans_for_testing) \
                       and not did_child_achieve_subgoal:
                        f_trans_testing = deepcopy(f_trans_0)
                        f_trans_testing.reward = self._child_failure_penalty
                        if not self._bootstrap_testing_transitions:
                            # Have to set done to True in these transitions 
                            # (according to HAC paper and accompanying code).
                            f_trans_testing.done = True
                        if self._verbose:
                            print("testing transition in {}\n".format(self.name) 
                                    + str(f_trans_testing))
                        self._add_to_flat_replay_buffer(f_trans_testing)

                    # hindsight action transition
                    # If child node achieved desired goal use original action.
                    # If not, use subgoal the child achieved as action.
                    f_trans_hindsight_action = deepcopy(f_trans_0)
                    if not did_child_achieve_subgoal:
                        f_trans_hindsight_action.action = deepcopy(f_trans_hindsight_action.action)
                        f_trans_hindsight_action.action["goal"] = tr.child_feedback["achieved_generalized_goal"]["goal"]
                        # TODO: In principle, reward could depend on action, so have to recompute
                        if self._verbose:
                            print("hindsight action transition in {}\n".format(self.name) 
                                    + str(f_trans_hindsight_action))
                    self._add_to_flat_replay_buffer(f_trans_hindsight_action)
                    f_trans_base = f_trans_hindsight_action

                # hindsight goal transitions (based on hindsight action 
                # transition or original transition in case of a sink node)
                achieved_timed_goals = self._sample_achieved_timed_goals(trans_index, parent_info.step)
                for hindsight_tg in achieved_timed_goals:
                    f_trans_hindsight_goal = deepcopy(f_trans_base)
                    f_trans_hindsight_goal.obs = {
                            "partial_observation": tr.subtask_tr.obs["partial_observation"], 
                            "desired_goal": hindsight_tg["desired_goal"], 
                            "delta_t_ach": hindsight_tg["delta_t_ach_obs"]}
                    f_trans_hindsight_goal.new_obs = {
                            "partial_observation": tr.subtask_tr.new_obs["partial_observation"], 
                            "desired_goal": hindsight_tg["desired_goal"], 
                            "delta_t_ach": hindsight_tg["delta_t_ach_new_obs"]}
                    h_tg = TimedGoal(
                            goal = hindsight_tg["desired_goal"], 
                            delta_t_ach = hindsight_tg["delta_t_ach_new_obs"][0], 
                            delta_t_comm = hindsight_tg["delta_t_ach_new_obs"][0])
                    achieved, reward, ach_time_up, _ = self._check_status(
                            achieved_goal = tr.subtask_tr.info["achieved_generalized_goal"]["goal"], 
                            desired_tg = h_tg, 
                            obs = f_trans_hindsight_goal.obs, 
                            action = f_trans_hindsight_goal.action, 
                            parent_info = parent_info,
                            env_info = tr.env_info) 
                    f_trans_hindsight_goal.done = ach_time_up
                    f_trans_hindsight_goal.reward = reward
                    self._add_to_flat_replay_buffer(f_trans_hindsight_goal)

        self._episode_transitions.clear()

    def get_algo_info(self, env_obs, parent_info):
        # child_be_deterministic instructs the children to be deterministic whereas
        # is_deterministic implies that this node is supposed to be deterministic
        if parent_info is not None:
            is_deterministic = parent_info.algo_info["child_be_deterministic"]
            child_be_deterministic = is_deterministic
        else:
            is_deterministic = False
            child_be_deterministic = False

        # if child_be_deterministic is true, all children have to use deterministic policies
        # until this node or an active parent gets back control (testing transition)
        child_be_deterministic = child_be_deterministic or np.random.rand() < self._testing_fraction
        new_algo_info = {
                "is_deterministic": is_deterministic, 
                "child_be_deterministic": child_be_deterministic
                }
        return new_algo_info
