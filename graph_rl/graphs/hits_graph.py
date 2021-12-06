import numpy as np

from . import Graph
from ..nodes import HiTSNode, HACNode
from ..utils import listify
from ..aux_rewards import EnvReward

class HiTSGraph(Graph):

    def __init__(self, name, n_layers, env, subtask_specs, HAC_kwargs, HiTS_kwargs, 
            update_sgs_rendering=None, update_tsgs_rendering=None, env_reward_weight=None):
        """
        Args:
        n_layers (int): Number of layers in hierarchy.
        """

        self.n_layers = n_layers
        self._update_sgs_rendering = update_sgs_rendering 
        self._update_tsgs_rendering = update_tsgs_rendering 

        # convert parameters to list with one entry per layer if not already a list
        HiTS_kwargs_list = listify(HiTS_kwargs, n_layers - 1)
        subtask_specs = listify(subtask_specs, n_layers - 1)

        # by default penalty for choosing a goal which was not reached by the child is minus
        # maximum number of actions in episode on this layer
        if "child_failure_penalty" not in HAC_kwargs:
            HAC_kwargs["child_failure_penalty"] = -subtask_specs[-1].max_n_actions
        for kwargs, subtask_spec in zip(HiTS_kwargs_list, subtask_specs):
            if "child_failure_penalty" not in kwargs:
                kwargs["child_failure_penalty"] = -1.

        # let higest level see environment reward in addition to penalty
        # for emitting timed subgoals if desired
        if env_reward_weight is not None:
            subtask_specs[-1].add_aux_reward(EnvReward(env_reward_weight))

        # highest node is HAC node because most environments do not provide a 
        # timed subgoal
        highest_node = HACNode(
                    name = "hac_node_layer_{}".format(n_layers - 1), 
                    parents = [],
                    subtask_spec = subtask_specs[-1], 
                    HAC_kwargs = HAC_kwargs)

        # construct graph
        parents = [highest_node]
        self._nodes = [highest_node]
        for l in range(n_layers - 2, -1, -1):
            node_name = "hits_node_layer_{}".format(l)
            node = HiTSNode(
                    name = node_name, 
                    parents = parents,
                    subtask_spec = subtask_specs[l], 
                    HiTS_kwargs = HiTS_kwargs_list[l])
            if l < n_layers - 1:
                self._nodes[0].add_child(node)
            self._nodes.insert(0, node)
            parents = [node]

        entry_node = self._nodes[-1]


        super(HiTSGraph, self).__init__(name, entry_node, env.action_space)

    def get_atomic_action(self, env_obs, sess_info, start_node = None, testing = False):
        action = super().get_atomic_action(env_obs, sess_info, start_node, testing)

        # update timed subgoals via callback for rendering them in environment
        if self._update_tsgs_rendering is not None:
            subgoals = [nd.current_timed_goal for nd in self._nodes[:-1]]
            tolerances = [nd.current_goal_tol for nd in self._nodes[:-1]]
            self._update_tsgs_rendering(subgoals, tolerances)

        return action

