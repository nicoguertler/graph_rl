from . import Node
from ..subtasks import TimedGoalSubtask, DictInfoHidingTolTGSubtaskSpec
from ..algorithms import HiTS

class HiTSNode(Node):

    def __init__(self, name, parents, subtask_spec, HiTS_kwargs, tb_writer = None):

        # create subtask
        subtask = TimedGoalSubtask(name + "_subtask", subtask_spec)

        # create algorithm
        algorithm = HiTS(
                name = name + "_algorithm",
                check_status = subtask.check_status, 
                convert_time = subtask.task_spec.convert_time, 
                delta_t_min = subtask.task_spec._delta_t_min,
                use_normal_trans_for_testing = False, 
                use_testing_transitions = True,  
                learn_from_deterministic_episodes = True, 
                **HiTS_kwargs)

        # policy creation is done via algorithm because it does the sampling part
        # of generating an action
        policy_class = algorithm.get_policy

        self.current_timed_goal = None
        self.current_goal_tol = None

        super(HiTSNode, self).__init__(name, policy_class, subtask, algorithm, parents)

    def check_parent_interruption(self, env_info, child_feedback, sess_info, 
            check_this_node = False):
        parent_info = self._parent_info[-1]
        self.current_timed_goal = self.subtask.get_updated_desired_tg_in_steps(env_info.new_obs, 
                parent_info, sess_info)
        if isinstance(self.subtask.task_spec, DictInfoHidingTolTGSubtaskSpec):
            self.current_goal_tol = parent_info.action["goal_tol"]
        elif hasattr(self.subtask.task_spec, "_goal_achievement_threshold"):
            self.current_goal_tol = self.subtask.task_spec._goal_achievement_threshold
        else:
            self.current_goal_tol = None
        return super(HiTSNode, self).check_parent_interruption(env_info, child_feedback, 
                sess_info, check_this_node)
