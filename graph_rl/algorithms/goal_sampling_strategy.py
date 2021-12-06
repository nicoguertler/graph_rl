from abc import ABC, abstractmethod


class GoalSamplingStrategy(ABC):

    @abstractmethod
    def __call__(self, episode_transitions, n_hindsight_goals):
        """Return a list of indices of transitions from which goals are generated.

        Hindsight goals are constructed from the achieved goal attribute of the 
        selected transitions."""
        pass


