from copy import deepcopy

import numpy as np
import torch

from ..data import FlatTransition
from . import OffPolicyAlgorithm
from .goal_sampling_strategy import GoalSamplingStrategy

class HAC(OffPolicyAlgorithm):
    """Hierarchical Actor-Critic."""

    supported_goal_sampling_strategies = {"future", "episode", "final"}

    def __init__(self, name, model, child_failure_penalty, check_achievement, 
            flat_algo_kwargs=None, flat_algo_name="SAC", goal_sampling_strategy="future", 
            buffer_size=100000, testing_buffer_size=50000, batch_size=128, n_hindsight_goals=4, 
            testing_fraction=0.3, fully_random_fraction=0.1, bootstrap_testing_transitions=True, 
            use_normal_trans_for_testing=False, use_testing_transitions=True, 
            learn_from_deterministic_episodes=True, log_q_values=True, learning_starts=0, 
            bootstrap_end_of_episode=True, grad_steps_per_env_step=1):
            """
            Args:
                check_achievement (callable): Maps achieved_goal, 
                    desired_goal and parent_info to boolean achieved and float
                    reward indicating whether the achieved_goal satisfies the 
                    desired goal and what reward this implies.
                """

            super(HAC, self).__init__(name, flat_algo_name, model, fully_random_fraction, 
                    flat_algo_kwargs, buffer_size, batch_size, learning_starts,
                    grad_steps_per_env_step)


            assert goal_sampling_strategy in self.supported_goal_sampling_strategies\
                    or isinstance(goal_sampling_strategy, GoalSamplingStrategy), \
                    "Goal sampling strategy {} not supported.".format(goal_sampling_strategy)

            self._child_failure_penalty = child_failure_penalty
            self._goal_sampling_strategy = goal_sampling_strategy
            self._check_achievement = check_achievement
            self._n_hindsight_goals = n_hindsight_goals
            self._testing_fraction = testing_fraction
            self._bootstrap_testing_transitions = bootstrap_testing_transitions

            self._use_normal_trans_for_testing = use_normal_trans_for_testing
            self._use_testing_transitions = use_testing_transitions
            self._learn_from_deterministic_episodes = learn_from_deterministic_episodes
            self._log_q_values = log_q_values
            self._bootstrap_end_of_episode = bootstrap_end_of_episode

            self._verbose = False

    def _sample_achieved_goals(self, current_index):
        goals = []
        if self._n_hindsight_goals > 0:
            n_transitions = len(self._episode_transitions)
            if self._goal_sampling_strategy == "future":
                n_goals = min(self._n_hindsight_goals, 
                        n_transitions - current_index - 1)
                indices = np.random.randint(low = current_index + 1, high = n_transitions, 
                        size = n_goals)
            elif self._goal_sampling_strategy == "episode":
                n_goals = min(self._n_hindsight_goals, n_transitions)
                indices = np.random.randint(low = 0, high = n_transitions, size = n_goals)
            elif self._goal_sampling_strategy == "final":
                # only generate hindsight goal from final state if environment is done
                if self._episode_transitions[-1].env_done:
                    indices = [-1]
                else:
                    indices = []
            elif isinstance(self._goal_sampling_strategy, GoalSamplingStrategy):
                indices = self._goal_sampling_strategy(self._episode_transitions, 
                        self._n_hindsight_goals)

            for i in indices:
                goals.append(self._episode_transitions[i].subtask_tr.info["achieved_generalized_goal"])

        return goals

    def _add_experience_to_flat_algo(self, parent_info, deterministic_episode, node_is_sink, sess_info):
        if self._learn_from_deterministic_episodes or not deterministic_episode:
            self._ep_return = 0
            # add transitions of this episode to replay buffer of flat RL algorithm
            # (by applying hindsight goal and action manipulations)
            for trans_index, tr in enumerate(self._episode_transitions):
                has_achieved_now = tr.subtask_tr.info.get("has_achieved", False)

                if self._bootstrap_end_of_episode:
                    done = has_achieved_now
                else:
                    done = has_achieved_now or tr.env_info.done

                # unaltered flat transition
                f_trans_0 = FlatTransition(
                        obs = tr.subtask_tr.obs, 
                        action = tr.subtask_tr.action,
                        reward = tr.subtask_tr.reward,
                        new_obs = tr.subtask_tr.new_obs,
                        done = done)

                # if the node is a sink, do not attempt to manipulate action
                # in hindsight and add original transition to replay buffer
                if node_is_sink:
                    testing_transition = False
                    f_trans_base = f_trans_0
                    self._add_to_flat_replay_buffer(f_trans_0)
                    self._ep_return += f_trans_0.reward

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
                        self._ep_return += f_trans_testing.reward

                    # hindsight action transition
                    # If child node achieved desired goal use original action.
                    # If not, use subgoal the child achieved as action.
                    f_trans_hindsight_action = deepcopy(f_trans_0)
                    if not testing_transition or did_child_achieve_subgoal:
                        self._ep_return += f_trans_hindsight_action.reward
                    if not did_child_achieve_subgoal:
                        f_trans_hindsight_action.action = tr.child_feedback["achieved_generalized_goal"]
                        # TODO: In principle, reward could depend on action, so have to recompute
                        if self._verbose:
                            print("hindsight action transition in {}\n".format(self.name) 
                                    + str(f_trans_hindsight_action))
                    self._add_to_flat_replay_buffer(f_trans_hindsight_action)
                    f_trans_base = f_trans_hindsight_action

                # hindsight goal transitions (based on hindsight action 
                # transition or original transition in case of a sink node)
                achieved_goals = self._sample_achieved_goals(trans_index)
                for hindsight_goal in achieved_goals:
                    f_trans_hindsight_goal = deepcopy(f_trans_base)
                    f_trans_hindsight_goal.obs = {
                        "partial_observation": tr.subtask_tr.obs["partial_observation"], 
                        "desired_goal": hindsight_goal
                    }
                    f_trans_hindsight_goal.new_obs = {
                        "partial_observation": tr.subtask_tr.new_obs["partial_observation"], 
                        "desired_goal": hindsight_goal
                    }
                    achieved, f_reward = self._check_achievement(
                            achieved_goal = tr.subtask_tr.info["achieved_generalized_goal"], 
                            desired_goal = hindsight_goal, 
                            obs = f_trans_hindsight_goal.obs, 
                            action = f_trans_hindsight_goal.action, 
                            parent_info = parent_info,
                            env_info = tr.env_info) 
                    f_trans_hindsight_goal.done = achieved
                    f_trans_hindsight_goal.reward = f_reward
                    self._add_to_flat_replay_buffer(f_trans_hindsight_goal)

            # log undiscounted ep return to tensorboard (includes contribution 
            # from testing transitions but not from hindsight transitions!)
            # tensorboard logging
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(f"{self.name}/ep_return", self._ep_return, sess_info.total_step)

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
