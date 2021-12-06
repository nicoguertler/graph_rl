from abc import ABC, abstractmethod

from ..logging import read_csv_log

class Subtask(ABC):

    def __init__(self, name, subtask_spec):
        super().__init__()
        self.name = name
        self._tb_writer = None
        self._aux_rewards = subtask_spec.get_aux_rewards()

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def parent_action_space(self):
        pass

    @abstractmethod
    def reset(self):
        """Executed when node is reset."""
        pass

    @abstractmethod
    def get_observation(self, env_obs, parent_info):
        pass

    @abstractmethod
    def check_interruption(self, env_info, new_subtask_obs, parent_info, sess_info):
        for aux_rew in self._aux_rewards:
            aux_rew.update(env_info)

    @abstractmethod
    def evaluate_transition(self, env_obs, env_info, subtask_trans, parent_info, algo_info, sess_info):
        pass

    def get_aux_rewards(self, obs, action, env_reward):
        """Get sum of auxiliary rewards."""

        r = 0
        for aux_rew in self._aux_rewards:
            r += aux_rew(obs, action, env_reward)
        return r

    def add_aux_reward(self, aux_reward):
        self._aux_rewards.append(aux_reward)

    def set_tensorboard_writer(self, tb_writer):
        self._tb_writer = tb_writer

    def create_logfiles(self, logdir, append = False):
        pass

    @classmethod
    def read_logfiles(cls, logfiles):
        data = {}
        for name, path in logfiles.items():
            data[name] = read_csv_log(path)
        return data

    def get_logfiles(self):
        return None

