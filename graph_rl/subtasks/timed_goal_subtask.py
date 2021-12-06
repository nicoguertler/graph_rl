from copy import copy
import os

import numpy as np

from . import Subtask
from ..spaces import DictSpace, BoxSpace, space_from_gym_space
from ..data import SubtaskTransition
from ..utils import listify
from ..logging import CSVLogger

class TimedGoalSubtask(Subtask):
    """Subtask is to achieve a goal at a given point in time."""

    def __init__(self, name, subtask_spec):
            """Args:
                subtask_spec (TimedGoalSubtaskSpec): Specification of the timed goal subtask.
            """
            super(TimedGoalSubtask, self).__init__(name, subtask_spec)

            self.task_spec = subtask_spec

            self._observation_space = DictSpace({
                "partial_observation": subtask_spec.partial_obs_space, 
                "desired_goal": subtask_spec.goal_space,
                "delta_t_ach": BoxSpace(low = [-1.], high = [1.]),
                })

            self._n_actions_taken = 0

            self.logger_test = None
            self.logger_train = None
            self.logfiles = {}

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def parent_action_space(self):
        return self.task_spec.parent_action_space

    def reset(self):
        self._n_actions_taken = 0

    def get_observation(self, env_obs, parent_info, sess_info):
        partial_obs = self.task_spec.map_to_partial_obs(env_obs, parent_info)
        desired_tg = self.task_spec.get_desired_timed_goal(env_obs, parent_info)

        full_obs = {
                "partial_observation": partial_obs, 
                "desired_goal": desired_tg.goal, 
                # policy sees desired time until achievement relative to current time step
                "delta_t_ach": [self.task_spec.convert_time(self.task_spec.unconvert_time(desired_tg.delta_t_ach) - (sess_info.ep_step - parent_info.step))]
                }
        return full_obs


    def check_status(self, achieved_goal, desired_tg, obs, action, parent_info, env_info):
        """Assumes that the desired_tg has delta_t attributes in [-1, 1] space and not in units of env steps."""
        ach_time_up = self.task_spec.unconvert_time(desired_tg.delta_t_ach) <= 0.
        # timed goal can only be achieved when the time is up
        if  ach_time_up and \
        self.task_spec.goal_achievement_criterion(achieved_goal, desired_tg.goal, parent_info):
            reward = 1.
            achieved = True
        else:
            reward = 0.             
            achieved = False
        # running out of commitment time 
        # NOTE: This is always false if commitment time is NaN
        comm_time_up = self.task_spec.unconvert_time(desired_tg.delta_t_comm) <= 0.

        # add auxiliary rewards
        if obs is not None and action is not None:
            reward += self.get_aux_rewards(obs, action, env_info.reward)
            
        return achieved, reward, ach_time_up, comm_time_up

    def _check_status_convenience(self, achieved_goal, obs, action, parent_info, sess_info, env_info):
        """Convenience version of check status also does the conversion of delta_t into [-1, 1] interval. 
        
        It assumes that the delta_t in parent_info has not been updated since the emission of the parent_info by 
        the parent. Hence, the elapsed time is subtracted from this outdated delta_t."""
        desired_tg = self.get_updated_desired_tg_in_steps(env_info.new_obs, parent_info, sess_info)
        desired_tg.delta_t_ach = self.task_spec.convert_time(desired_tg.delta_t_ach)
        desired_tg.delta_t_comm = self.task_spec.convert_time(desired_tg.delta_t_comm)
        
        return self.check_status(achieved_goal, desired_tg, obs, action, parent_info, env_info)


    def check_interruption(self, env_info, new_subtask_obs, parent_info, sess_info):
        super().check_interruption(env_info, new_subtask_obs, parent_info, sess_info)
        new_env_obs = env_info.new_obs
        new_partial_obs = self.task_spec.map_to_partial_obs(new_env_obs, parent_info)
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        _, _, ach_time_up, comm_time_up = self._check_status_convenience(
                achieved_goal, None, None, parent_info, sess_info, env_info)
        return ach_time_up or comm_time_up

    def evaluate_transition(self, env_obs, env_info, subtask_trans, parent_info, algo_info, sess_info):
        new_partial_obs = self.task_spec.map_to_partial_obs(env_info.new_obs, parent_info)
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        achieved, reward, ach_time_up, comm_time_up = self._check_status_convenience(achieved_goal, 
                subtask_trans.obs, subtask_trans.action, parent_info, sess_info, env_info)

        # subtask ended if subgoal achieved or running out of commitment time
        ended = ach_time_up or comm_time_up

        self._n_actions_taken += 1
        if ended:
            n_actions_taken = self._n_actions_taken
            self._n_actions_taken = 0

        # sample delta_t_ach for achieved timed goal from uniform distribution over times that would have run out
        # in this env step
        achieved_timed_goal = self.task_spec.get_achieved_timed_goal_dict(
                achieved_goal = achieved_goal, 
                delta_t_ach = sess_info.ep_step - parent_info.step - np.random.rand(), 
                parent_info = parent_info)

        # NOTE: The boolean subtask_ended only indicates whether the subtask ended, not that the goal was reached!
        # Whether the goal was reached is encoded in the key "has_achieved" in info.
        info = {
                "has_achieved": achieved,
                "comm_time_up": comm_time_up, 
                "ach_time_up": ach_time_up, 
                "achieved_generalized_goal": achieved_timed_goal
                }
        feedback = copy(info)
        #info["delta_t_comm"] = desired_tg.delta_t_comm - (sess_info.ep_step - parent_info.step)
        info["t"] = sess_info.ep_step

        complete_subtask_trans = copy(subtask_trans)
        complete_subtask_trans.reward = reward
        complete_subtask_trans.ended = ended
        complete_subtask_trans.info = info

        # tensorboard logging
        if self._tb_writer is not None and ach_time_up:
                mode = "test" if sess_info.testing else "train"
                self._tb_writer.add_scalar(f"{self.name}/{mode}/subgoal_achieved", int(achieved), sess_info.total_step)
                self._tb_writer.add_scalar(f"{self.name}/{mode}/n_actions", n_actions_taken, sess_info.total_step)
        # csv logging
        if ach_time_up and self.logger_train is not None and self.logger_test is not None:
            row_dict = {
                    "achieved": int(achieved), 
                    "n_actions": n_actions_taken, 
                    "step": sess_info.total_step, 
                    "time": self.logger_test.time_passed() if sess_info.testing else self.logger_train.time_passed()
                    }
            if sess_info.testing:
                self.logger_test.log(row_dict)
            elif self.logger_train:
                self.logger_train.log(row_dict)

        return complete_subtask_trans, feedback
    
    def get_updated_desired_tg_in_steps(self, env_obs, parent_info, sess_info):
        """Returns desired timed goal in global time (i.e. env steps)."""
        desired_tg = self.task_spec.get_desired_timed_goal(env_obs, parent_info)

        elapsed_time = sess_info.ep_step - parent_info.step

        desired_tg.delta_t_ach = self.task_spec.unconvert_time(desired_tg.delta_t_ach) - elapsed_time
        desired_tg.delta_t_comm = self.task_spec.unconvert_time(desired_tg.delta_t_comm) - elapsed_time

        return desired_tg

    def create_logfiles(self, logdir, append):
        if logdir is not None:
            logfile_test = os.path.join(logdir, self.name  + "_test.csv")
            logfile_train = os.path.join(logdir, self.name  + "_train.csv")
            self.logger_test = CSVLogger(logfile_test, ("achieved", "n_actions", "step", "time"), append)
            self.logger_train = CSVLogger(logfile_train, ("achieved", "n_actions", "step", "time"), append)
            self.logfiles["test"] = logfile_test
            self.logfiles["train"] = logfile_train

    def get_logfiles(self):
        return self.logfiles


