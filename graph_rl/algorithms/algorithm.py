from abc import ABC, abstractmethod

class Algorithm(ABC):
    """Algorithm base class."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self._tb_writer = None

    @abstractmethod
    def add_experience(self, env_obs, env_info, subtask_trans, parent_info, 
            child_feedback, algo_info, interruption, sess_info):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass

    def get_algo_info(self, env_obs, parent_info):
        return None

    def set_tensorboard_writer(self, tb_writer):
        self._tb_writer = tb_writer

    @abstractmethod
    def get_parameters(self):
        """Returns dictionary containing all learned parameters."""
        pass

    @abstractmethod
    def get_state(self):
        """Returns dictionary containing state."""
        pass

    @abstractmethod
    def save_state(self, node_id):
        """Saves state and returns dictionary with file paths."""
        pass

    @abstractmethod
    def load_parameters(self, params):
        """Load all learned parameters from dictionary."""
        pass

    @abstractmethod
    def load_state(self, state, dir_path = None):
        """Load state from dictionary."""
        pass

