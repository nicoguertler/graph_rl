import os
import torch

from ..nodes import Node
from ..data import FlatTransition

class Graph:
    """Graph defining the hierarchy.

    The attribute entry_node contains the root node of the graph while 
    active_node keeps track of which node is currently active.
    """

    def __init__(self, name, entry_node, env_action_space, global_algorithms = None):
        self.name = name
        self.entry_node = entry_node
        self.active_node = self.entry_node
        self.env_action_space = env_action_space
        self.global_algorithms = global_algorithms

        self._env_obs = None
        self._atomic_action = None
        self._env_return = 0
        self.tb_writer = None

        # create policies in the nodes recursively
        self.entry_node.create_policy(env_action_space)
        self.reset()

    def reset(self):
        self.active_node = self.entry_node
        self.active_parent = None
        self.active_parent_info = None

        self.entry_node.reset()

    def get_atomic_action(self, env_obs, sess_info, start_node = None, testing = False):
        self._env_obs = env_obs
        
        # forward pass through graph starting from active node/provided start node
        if start_node is not None:
            self.active_node = start_node

        assert self.active_node is not None

        action, self.active_node = self.active_node.get_action(
                env_obs = env_obs, 
                parent_info = self.active_parent_info, 
                parent = self.active_parent,
                sess_info = sess_info, 
                testing = testing)
        self._atomic_action = action

        return action

    def register_experience(self, env_info, sess_info):
        # First backward path starting from active node: check for interruption by parent
        parent_interruption, interrupting_node = self.active_node.check_parent_interruption(
                env_info, Node.sink_identification_feedback, sess_info)

        # Second backward pass through graph starting from active node:
        # Propagate feedback from children back to parents and register 
        # experience in nodes that acted.         
        self.active_node, self.active_parent, self.active_parent_info = \
                self.active_node.register_experience(env_info, Node.sink_identification_feedback, 
                        sess_info, interrupting_node)

        # global algorithms
        if self.global_algorithms:
            trans = FlatTransition(self._env_obs, self._atomic_action, env_info)
            for g_algo in self.global_algorithms:
                g_algo.add_experience(trans, sess_info)
                if learn:
                    g_algo.learn(sess_info)

        # is execution of graph finished?
        return self.active_node is None

    def learn(self, sess_info):
        """Learn without doing a rollout.

        Can be used for learning after instead of during a rollout."""
        nodes = self.entry_node.get_descendants_recursively()
        for node in nodes:
            node.algorithm.learn(sess_info.total_step)

    def set_tensorboard_writer(self, tb_writer):
        self._tb_writer = tb_writer
        self.entry_node.set_tensorboard_writer(tb_writer, [])

    def create_logfiles(self, logdir, append = False):
        nodes = self.entry_node.get_descendants_recursively()
        for node in nodes:
            node.create_logfiles(logdir, append)

    def get_logfiles(self):
        nodes = self.entry_node.get_children_recursively()
        logfile_dict = {}
        for node in nodes:
            logfile_dict.update(node.get_logfiles())
        return logfile_dict

    def get_node_ids(self):
        nodes = self.entry_node.get_descendants_recursively()
        node_id_dict = {}
        ID = 0
        for n in nodes:
            node_id_dict[n] = ID
            ID += 1
        return node_id_dict

    def get_parameters(self):
        """Returns dictionary containing all learned parameters in this graph."""
        return self.entry_node.get_parameters_recursively()

    def get_state(self):
        """Returns dictionary containing state of this graph.

        'State' refers to properties which influence training but not execution of the graph.
        This includes the state of the optimizers, replay buffers etc."""
        return self.entry_node.get_state_recursively()

    def load_parameters(self, params):
        """Load all learned parameters in this graph from file or dictionary."""
        if isinstance(params, str):
            params = torch.load(params)
        assert isinstance(params, dict)
        self.entry_node.load_parameters_recursively(params)

    def load_state(self, state = None, dir_path = None):
        """Load state of this graph from files in given directory or from dict."""
        assert {state, dir_path} != {None, None}, "Either state dict or directory with state files has to be specified."
        if state is not None:
            assert isinstance(state, dict)
            self.entry_node.load_state_recursively(state)
        else:
            s = torch.load(os.path.join(dir_path, "graph_state.pt"))
            self.entry_node.load_state_recursively(s, dir_path)

    def save_parameters(self, path):
        """Save all learned parameters in this graph to file."""
        params = self.get_parameters()
        torch.save(params, path)

    def save_state_pickle(self, path):
        """Save state of this graph to file using pickle/torch.save.
        
        'State' includes the state of the optimizers, replay buffers, 
        model parameters etc."""
        state = self.get_state()
        torch.save(state, path)

    def save_state(self, dir_path):
        """Save state of this graph to files in given directory.
        
        'State' includes the state of the optimizers, replay buffers, 
        model parameters etc."""
        node_ids = self.get_node_ids()
        state = self.entry_node.save_state_recursively(node_ids, dir_path)
        torch.save(state, os.path.join(dir_path, "graph_state.pt"))


