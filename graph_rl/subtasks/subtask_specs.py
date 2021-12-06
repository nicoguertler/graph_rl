from abc import ABC, abstractmethod

class SubtaskSpec(ABC):
    """Base class for subtask specifications."""

    def __init__(self):
        super().__init__()
        self._aux_rewards = []

    def add_aux_reward(self, aux_reward):
        """Add auxiliary reward."""
        self._aux_rewards.append(aux_reward)

    def get_aux_rewards(self):
        """Get list of auxiliary rewards."""
        return self._aux_rewards
