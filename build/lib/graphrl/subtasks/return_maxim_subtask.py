from copy import copy

import numpy as np

from . import Subtask


class ReturnMaximSubtask(Subtask):
    """Subtask is to maximize expected return."""

    def __init__(self, name, subtask_spec):
        """Args:
            subtask_spec (ReturnMaximSubtaskSpec): Specification of the return maximization subtask.
        """

        super().__init__(name, subtask_spec)
        self.task_spec = subtask_spec

        self._n_actions_taken = 0
        self._return = 0
        #self._min = np.array([np.inf]*8)
        #self._max = np.array([-np.inf]*8)

    @property
    def observation_space(self):
        return self.task_spec.obs_space

    @property
    def parent_action_space(self):
        return self.task_spec.parent_action_space

    def reset(self):
        self._n_actions_taken = 0
        self._return = 0
        self.task_spec.reset()

    def get_observation(self, env_obs, parent_info, sess_info):
        return self.task_spec.map_to_obs(env_obs, parent_info, sess_info.ep_step)

    def check_interruption(self, env_info, new_subtask_obs, parent_info, sess_info):
        # update internal state of subtask spec in each time step
        self.task_spec.step_update(env_info, parent_info, sess_info)
        # return maximization subtask does not interrupt child nodes
        return False

    def evaluate_transition(self, env_obs, env_info, subtask_trans, parent_info, algo_info, sess_info):
        new_obs = self.task_spec.map_to_obs(
            env_info.new_obs, parent_info, sess_info.ep_step
        )
        #self._min = np.minimum(self._min, new_obs)
        #self._max = np.maximum(self._max, new_obs)
        #print(self._max)
        reward = self.task_spec.get_reward(
            subtask_trans.obs, subtask_trans.action, new_obs, parent_info, 
            sess_info.ep_step, env_info
        )
        # add auxiliary rewards
        if new_obs is not None and subtask_trans.action is not None:
            reward += self.get_aux_rewards(new_obs, subtask_trans.action)
        self._n_actions_taken += 1
        self._return += reward

        # subtask ended if running out of actions
        if (self.task_spec.max_n_actions is not None and 
            self._n_actions_taken >= self.task_spec.max_n_actions):
            # tensorboard logging
            if self._tb_writer is not None:
                    mode = "test" if sess_info.testing else "train"
                    self._tb_writer.add_scalar(f"{self.name}/{mode}/return", self._return, sess_info.total_step)
            self._return = 0
            self._n_actions_taken = 0
            ended = True
        else:
            ended = False

        info = {}
        feedback = {}

        complete_subtask_trans = copy(subtask_trans)
        complete_subtask_trans.reward = reward
        complete_subtask_trans.ended = ended
        complete_subtask_trans.info = info
        

        return complete_subtask_trans, feedback
