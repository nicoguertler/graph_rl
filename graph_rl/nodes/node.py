import os

from ..spaces import space_from_gym_space
from ..data import SubtaskTransition, ParentInfo

class Node:
    """Node in the directed graph defining the hierarchy.

    Contains a policy, a subtask and an algorithm."""

    sink_identification_feedback = object()

    def __init__(self, name, policy_class, subtask, algorithm, parents, 
            interruption_policy = None):
        self.name = name
        self._policy_class = policy_class
        self.policy = None
        self.subtask = subtask
        self.algorithm = algorithm
        self.tb_writer = None
        if interruption_policy is None:
            self.interruption_policy = lambda env_obs, subtask_obs: False
        else:
            self.interruption_policy = interruption_policy

        self._children = []
        self._parents = parents

        self._env_obs = []
        self._parent_info = []
        self._active_parent = []
        self._ind_child_to_activate = []
        self._action = []
        self._algo_info = []
        self._env_observation = []
        self._subtask_obs = []

        self._is_reset = True

    def reset(self):
        if self._is_reset == False:
            self._env_obs.clear()
            self._parent_info.clear()
            self._active_parent.clear()
            self._ind_child_to_activate.clear()
            self._action.clear()
            self._algo_info.clear()
            self._env_observation.clear()
            self._subtask_obs.clear()

            self.subtask.reset()

            self._is_reset = True

            for child in self._children:
                child.reset()

    def add_child(self, child):
        self._children.append(child)

    def add_parent(self, parent):
        self._parents.append(parent)

    def create_policy(self, env_action_space):
        # have to know all parents and children at this point
        if self.policy is None:
            p_name = self.name + "_policy"

            observation_space = self.subtask.observation_space

            if not self._children:
                action_space = space_from_gym_space(env_action_space)
            else:
                child_parent_action_space = self._children[0].subtask.parent_action_space
                assert all([child.subtask.parent_action_space == child_parent_action_space 
                    for child in self._children[1:]])
                action_space = child_parent_action_space

            self.policy = self._policy_class(p_name, observation_space, action_space)

            for child in self._children:
                child.create_policy(env_action_space)

    def get_action(self, env_obs, parent_info, parent, sess_info, testing = False):
        self._is_reset = False

        # use subtask to map environment observation and and parent information 
        # to observation for this node 
        subtask_obs = self.subtask.get_observation(env_obs, parent_info, sess_info)

        algo_info = self.algorithm.get_algo_info(env_obs, parent_info)

        ind_child_to_activate, action = self.policy(subtask_obs, algo_info, testing = testing)

        self._ind_child_to_activate.append(ind_child_to_activate)

        # keep track of env obs, active parent, parent info etc. in order to be able
        # register this experience for learning later on
        self._env_obs.append(env_obs)
        self._active_parent.append(parent)
        self._parent_info.append(parent_info)
        self._subtask_obs.append(subtask_obs)
        self._algo_info.append(algo_info)
        self._action.append(action)

        # if node is a sink (no children), return output of policy as 
        # (atomic) action and self as active node
        if not self._children:
            return action, self
        # otherwise get action from child that is to be activated
        else:
            parent_info_for_child = ParentInfo(action, algo_info, sess_info.ep_step)
            return self._children[ind_child_to_activate].get_action(
                    env_obs, parent_info_for_child, self, sess_info, testing)

    def check_parent_interruption(self, env_info, child_feedback, sess_info, check_this_node = False):
        if self._parents:
            # check whether parents want to interrupt (have priority over children)
            self._interrupted_by_parent, interrupting_parent_node = \
            self._active_parent[-1].check_parent_interruption(env_info, child_feedback, 
                    sess_info, check_this_node = True)
        else:
            self._interrupted_by_parent = False

        # check for interruption in this node
        if check_this_node:
            parent_info = self._parent_info[-1]
            new_subtask_obs = self.subtask.get_observation(env_info.new_obs, parent_info, sess_info)

            # check whether the interruption policy wants to interrupt
            self._interrupted_by_pol = self.interruption_policy(env_info.new_obs, new_subtask_obs)

            # check whether the subtask wants to interrupt
            self._interrupted_by_subtask = self.subtask.check_interruption(env_info, 
                    new_subtask_obs, parent_info, sess_info)
        else:
            self._interrupted_by_pol = False
            self._interrupted_by_subtask = False

        # interruption by parent has priority over interruption by child
        if self._interrupted_by_parent:
            return True, interrupting_parent_node
        elif self._interrupted_by_pol or self._interrupted_by_subtask:
            return True, self
        else:
            return False, None


    def register_experience(self, env_info, child_feedback, sess_info, interrupting_node = None):
        env_obs = self._env_obs.pop()
        ind_child_to_activate = self._ind_child_to_activate.pop()
        action = self._action.pop()
        algo_info = self._algo_info.pop()

        active_parent = self._active_parent.pop()
        parent_info = self._parent_info.pop()
        subtask_obs = self._subtask_obs.pop()

        new_subtask_obs = self.subtask.get_observation(env_info.new_obs, parent_info, sess_info)

        # incomplete subtask transition, still missing reward, ended and info
        st_trans_incmplt = SubtaskTransition(subtask_obs, action, ind_child_to_activate, new_obs = new_subtask_obs)

        subtask_trans, node_feedback = self.subtask.evaluate_transition(env_obs, env_info, 
                st_trans_incmplt, parent_info, algo_info, sess_info)

        if not sess_info.testing:
            self.algorithm.add_experience(env_obs, env_info, subtask_trans, parent_info, 
                    child_feedback, algo_info, interrupting_node is not None, sess_info)

        # TODO: Evaluate whether learning during an episode could be a 
        # useful feature
        #if sess_info.learn and not sess_info.testing:
        #    self.algorithm.learn(sess_info.total_step)

        # Return control to parent node if:
        # - there is an interrupting node and it is not this node
        # - the subtask in this node has ended
        # - the environment is done
        if (interrupting_node not in [None, self] or subtask_trans.ended or 
                env_info.done):
            # Return control to active parent if there is one
            if active_parent is not None:
                # current node can't interrupt parents
                if interrupting_node is self:
                    interrupting_node = None
                return active_parent.register_experience(env_info, node_feedback, 
                        sess_info, interrupting_node)
            # If there is no active parent return None in order to signal termination of graph
            else:
                return None, None, None
        # else return this node as new active node
        else:
            return self, active_parent, parent_info

    def get_descendants_recursively(self, visited = None):
        if visited is None:
            visited = set()
        visited.add(self)
        children = [self]
        for child in self._children:
            if child not in visited:
                children.extend(child.get_descendants_recursively(visited))
        return children

    def set_tensorboard_writer(self, tb_writer, visited_nodes):
        self._tb_writer = tb_writer
        self.subtask.set_tensorboard_writer(tb_writer)
        self.algorithm.set_tensorboard_writer(tb_writer)
        visited_nodes.append(self)
        for child in self._children:
            if child not in visited_nodes:
                child.set_tensorboard_writer(tb_writer, visited_nodes)
    
    def create_logfiles(self, logdir, append = False):
        self.subtask.create_logfiles(logdir, append)

    def get_logfiles(self):
        subtask_logfiles = self.subtask.get_logfiles()
        logfile_dict = {self.name: {
            "type": self.__class__.__name__, 
            "subtask_logfiles": self.subtask_logfiles
            } }
        return logfile_dict

    def get_parameters(self):
        """Returns dictionary containing all learned parameters of this node only."""
        return self.algorithm.get_parameters()

    def get_state(self):
        """Returns dictionary containing state of this node only."""
        return self.algorithm.get_state()

    def save_state(self, node_id, dir_path):
        """Saves state of this node only and returns dict with paths to files."""
        return self.algorithm.save_state(node_id, dir_path)

    def load_parameters(self, params):
        """Load all learned parameters from dictionary of this node only."""
        self.algorithm.load_parameters(params)

    def load_state(self, state, dir_path = None):
        """Load state of this node only."""
        self.algorithm.load_state(state, dir_path = dir_path)

    def get_parameters_recursively(self, visited = None):
        """Returns dictionary containing all learned parameters of this node and its descendants.
        
        The dictionary is assembled recursively so that the parameters of all nodes which 
        are reachable from this node are contained in it."""
        if visited is None:
            visited = set()
        visited.add(self)
        children_params = []
        for child in self._children:
            if child in visited:
                child_params = None
            else:
                child_params = child.get_parameters_recursively(visited)
            children_params.append(child_params)
        params = {
                "node_params": self.get_parameters(),
                "children_params": children_params
                }
        return params

    def get_state_recursively(self, visited = None):
        """Returns dictionary containing the state of this node and its descendants.
        
        The dictionary is assembled recursively so that the state of all nodes which 
        are reachable from this node are contained in it."""
        if visited is None:
            visited = set()
        visited.add(self)
        children_state = []
        for child in self._children:
            if child in visited:
                child_state = None
            else:
                child_state = child.get_state_recursively(visited)
            children_state.append(child_state)
        state = {
                "node_state": self.get_state(),
                "children_state": children_state
                }
        return state

    def save_state_recursively(self, node_ids, dir_path, visited = None):
        """Returns dictionary containing the state of this node and its descendants.
        
        Note: In this version the state may contain paths to auxiliary files that have 
        to be taken into account when loading the state. The dictionary is assembled 
        recursively so that the state of all nodes which are reachable from this node 
        are contained in it."""
        if visited is None:
            visited = set()
        visited.add(self)
        children_state = []
        for child in self._children:
            if child in visited:
                child_state = None
            else:
                child_state = child.save_state_recursively(node_ids, dir_path, visited)
            children_state.append(child_state)
        state = {
                "node_state": self.save_state(node_ids[self], dir_path),
                "children_state": children_state
                }
        return state


    def load_parameters_recursively(self, params):
        """Load all learned parameters of this node and all nodes reachable from it."""
        assert len(self._children) == len(params["children_params"])
        for child, child_params in zip(self._children, params["children_params"]):
            if child_params is not None:
                child.load_parameters_recursively(child_params)
        self.load_parameters(params["node_params"])

    def load_state_recursively(self, state, dir_path = None):
        """Load state of this node and all nodes reachable from it."""
        assert len(self._children) == len(state["children_state"])
        for child, child_state in zip(self._children, state["children_state"]):
            if child_state is not None:
                child.load_state_recursively(child_state, dir_path)
        self.load_state(state["node_state"], dir_path)
