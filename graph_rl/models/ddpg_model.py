import torch

from .mlp_actor_critic_model import MLPActorCriticModel
from ..networks import DDPGNetwork

class DDPGModel(MLPActorCriticModel):
    def __init__(self, hidden_layers_actor, hidden_layers_critics, activation_fns_actor, 
            activation_fns_critics, learning_rate_actor, learning_rate_critics, 
            device, squash_critics = False, low = None, high = None, 
            critic_clip_threshold = None, actor_clip_threshold = None, force_negative = False):
        super(DDPGModel, self).__init__(hidden_layers_actor, hidden_layers_critics, 
                activation_fns_actor, activation_fns_critics, 1, learning_rate_actor, 
                learning_rate_critics, device, force_negative = force_negative)


    def create(self, state_dim, action_dim):
        super(DDPGModel, self).create(state_dim, action_dim)
        self._actor = DDPGNetwork(self._actor_stump, 1., self._hidden_layers_actor[-1], 
                action_dim, device = self._device)
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), 
                lr = self._learning_rate_actor)

    @property
    def actor(self):
        return self._actor

    @property
    def actor_optim(self):
        return self._actor_optim

    def get_parameters(self):
        """Returns dictionary containing all learned parameters."""
        params = {
                "actor_state_dict": self._actor.state_dict(), 
                "mlp_params": super(DDPGModel, self).get_parameters()
                }
        return params

    def get_state(self):
        """Returns dictionary containing state."""
        state = {
                "actor_optim_state_dict": self._actor_optim.state_dict(), 
                "mlp_state": super(DDPGModel, self).get_state()
                }
        return state

    def load_parameters(self, params):
        """Load all learned parameters from dictionary."""
        super(DDPGModel, self).load_parameters(params["mlp_params"])
        self._actor.load_state_dict(params["actor_state_dict"])

    def load_state(self, state):
        """Load state from dictionary."""
        super(DDPGModel, self).load_state(state["mlp_state"])
        self._actor_optim.load_state_dict(state["actor_optim_state_dict"])
