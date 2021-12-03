from . import Node
from ..subtasks import (
    ShortestPathSubtask, ShortestPathSubtaskSpec, ReturnMaximSubtask, ReturnMaximSubtaskSpec
)
from ..algorithms import HAC

class HACNode(Node):

    def __init__(self, name, parents, subtask_spec, HAC_kwargs, tb_writer=None, interruption_policy=None):

        # create subtask (shortest path and reward maximization subtasks are supported)
        if isinstance(subtask_spec, ShortestPathSubtaskSpec):
            subtask = ShortestPathSubtask(name + "_subtask", subtask_spec)
        elif isinstance(subtask_spec, ReturnMaximSubtaskSpec):
            subtask = ReturnMaximSubtask(name + "_subtask", subtask_spec)
        else:
            raise NotImplementedError("An HAC node supports only a shortest path and a return maximization subtask.")

        # create algorithm
        check_achievement = subtask.check_achievement if hasattr(subtask, "check_achievement") else None 
        algorithm = HAC(
                name = name + "_algorithm",
                check_achievement = check_achievement,
                use_normal_trans_for_testing = False,
                use_testing_transitions = True,  
                learn_from_deterministic_episodes = True, 
                **HAC_kwargs)

        # policy creation is done via algorithm because it does the sampling part
        # of generating an action
        policy_class = algorithm.get_policy

        super(HACNode, self).__init__(name, policy_class, subtask, algorithm, parents, interruption_policy)

    def get_action(self, env_obs, parent_info, parent, sess_info, testing):
        if hasattr(self.subtask.task_spec, "get_desired_goal"):
            self.current_goal = self.subtask.task_spec.get_desired_goal(env_obs, parent_info)
        return super(HACNode, self).get_action(env_obs, parent_info, parent, sess_info, testing)
