import numpy as np

from . import Graph
from ..subtasks import EnvSPSubtaskSpec
from ..nodes import HACNode
from ..utils import listify

class HACGraph(Graph):

    def __init__(self, name, n_layers, env, subtask_specs, HAC_kwargs, update_sgs_rendering=None):
        """
        Args:
        n_layers (int): Number of layers in hierarchy.
        """

        self.n_layers = n_layers
        self._update_sgs_rendering = update_sgs_rendering 

        # convert HAC parameters to list with one entry per layer if not already a list
        HAC_kwargs_list = listify(HAC_kwargs, n_layers)
        subtask_specs = listify(subtask_specs, n_layers - 1)

        # by default penalty for choosing a goal which was not reached by the child is minus
        # maximum number of actions in episode on this layer
        for kwargs, subtask_spec in zip(HAC_kwargs_list, subtask_specs):
            if "child_failure_penalty" not in kwargs:
                kwargs["child_failure_penalty"] = -subtask_spec.max_n_actions

        # make sure testing fraction in lowest level is 0
        HAC_kwargs_list[0]["testing_fraction"] = 0.

        # construct graph
        parents = []
        self._nodes = []
        for l in range(n_layers - 1, -1, -1):
            node_name = "hac_node_layer_{}".format(l)
            node = HACNode(
                    name=node_name, 
                    parents=parents,
                    subtask_spec=subtask_specs[l], 
                    HAC_kwargs=HAC_kwargs_list[l])
            if l < n_layers - 1:
                self._nodes[0].add_child(node)
            self._nodes.insert(0, node)
            parents = [node]

        entry_node = self._nodes[-1]

        super(HACGraph, self).__init__(name, entry_node, env.action_space)

    def get_atomic_action(self, env_obs, sess_info, start_node=None, testing=False):
        action = super().get_atomic_action(env_obs, sess_info, start_node, testing)

        # update subgoals via callback for rendering them in environment
        if len(self._nodes) > 1 and self._update_sgs_rendering is not None:
            self._update_sgs_rendering([nd.current_goal for nd 
                in self._nodes[:-1]])

        return action

