from abc import ABC, abstractmethod

class Model:
    @abstractmethod
    def create(self, state_dim, action_dim):
        pass

    @property
    @abstractmethod
    def actor(self):
        pass

    @property
    @abstractmethod
    def actor_optim(self):
        pass

    @property
    def critics(self):
        return None

    @property
    def critic_optims(self):
        return None

    @abstractmethod
    def get_parameters(self):
        """Returns dictionary containing all learned parameters."""
        pass

    @abstractmethod
    def get_state(self):
        """Returns dictionary containing state."""
        pass

    @abstractmethod
    def load_parameters(self, params):
        """Load all learned parameters from dictionary."""
        pass

