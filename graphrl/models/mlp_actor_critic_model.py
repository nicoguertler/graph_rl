from abc import ABC, abstractmethod

import torch

from .model import Model
from ..networks import MLPNetwork, ConcatNetwork, SquashNetwork, NegativeNetwork

class MLPActorCriticModel(Model):
    def __init__(self, hidden_layers_actor, hidden_layers_critics, activation_fns_actor, 
            activation_fns_critics, n_critics, learning_rate_actor, learning_rate_critics, 
            device, squash_critics = False, low = None, high = None, force_negative = False):
        self._hidden_layers_actor = hidden_layers_actor
        self._hidden_layers_critics = hidden_layers_critics
        self._activation_fns_actor = activation_fns_actor
        self._activation_fns_critics = activation_fns_critics
        self._n_critics = n_critics
        self._learning_rate_actor = learning_rate_actor
        self._learning_rate_critics = learning_rate_critics
        self._device = device
        self._squash_critics = squash_critics
        self._low = low
        self._high = high
        self._force_negative = force_negative

    @abstractmethod
    def create(self, state_dim, action_dim):
        self._actor_stump = MLPNetwork(state_dim, self._hidden_layers_actor, self._activation_fns_actor, 
                device = self._device)

        self._critics = []
        self._critic_optims = []
        for _ in range(self._n_critics):
            critic = MLPNetwork(state_dim + action_dim, self._hidden_layers_critics + [1], 
                    self._activation_fns_critics + [None])
            if self._force_negative:
                critic = NegativeNetwork(critic, self._device)
            elif self._squash_critics:
                critic = SquashNetwork(critic, self._low, self._high, self._device)
            critic = ConcatNetwork(critic, device = self._device)
            self._critics.append(critic)
            self._critic_optims.append(torch.optim.Adam(critic.parameters(), lr = self._learning_rate_critics))

    def set_up_clipping(self, actor_threshold, critic_threshold):
        """Set up hooks for gradient clipping."""

        def rescale(grad, threshold):
            norm = torch.norm(grad)
            if norm > threshold:
                return threshold*grad/norm
            else:
                return grad

        if critic_threshold is not None:
            for network in self._critics:
                for p in network.parameters():
                    p.register_hook(lambda grad: rescale(grad, critic_threshold))

        if actor_threshold is not None:
            for p in self._actor.parameters():
                    p.register_hook(lambda grad: rescale(grad, actor_threshold))


    def set_learning_rate_critics(self, lr):
        self._learning_rate_critics = lr
        for optim in self._critic_optims:
            for g in optim.param_groups:
                g["lr"] = lr

    def set_learning_rate_actor(self, lr):
        self._learning_rate_actor = lr
        for g in self._actor_optim.param_groups:
            g["lr"] = lr

    @property
    def actor(self):
        return self._actor


    @property
    def critics(self):
        return self._critics


    @property
    def critic_optims(self):
        return self._critic_optims


    def get_parameters(self):
        """Returns dictionary containing all learned parameters."""
        # actors will be stored in derived classes
        params = {
                "critic_state_dicts": [cr.state_dict() for cr in self._critics]
                }
        return params


    def get_state(self):
        """Returns dictionary containing state."""
        state = {
                "critic_optim_state_dicts": [cr_optim.state_dict() for cr_optim in self._critic_optims]
                }
        return state


    def load_parameters(self, params):
        """Load all learned parameters from dictionary."""
        for critic, state_dict in zip(self._critics, params["critic_state_dicts"]):
            critic.load_state_dict(state_dict)


    def load_state(self, state):
        """Load state from dictionary."""
        for critic_optim, state_dict in zip(self._critic_optims, state["critic_optim_state_dicts"]):
            critic_optim.load_state_dict(state_dict)

