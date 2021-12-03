import numpy as np
import torch

from tianshou.data import Batch

from . import Policy
from ..networks import SACNetwork, DDPGNetwork

class TianshouPolicy(Policy):

    def __init__(self, name, observation_space, action_space, ts_policy, hidden_net = None, fully_random_fraction = 0.):
        """
        Args: 
            ts_policy: Tianshou policy. 
            hidden_net: Pytorch module returning either (mu, sigma), hidden_state for SAC or just action, hidden_state for DDPG.
        """
        super(TianshouPolicy, self).__init__(name, observation_space, action_space)
        self._fully_random_fraction = fully_random_fraction
        self._ts_policy = ts_policy
        self._hidden_net = hidden_net
        flat_action_space = self.action_space.get_flat_space()
        self._action_scale = 0.5*(flat_action_space.high - flat_action_space.low)
        self._action_bias = 0.5*(flat_action_space.low + flat_action_space.high)

    def __call__(self, observation, algo_info, testing = False):
        if testing:
            deterministic = True
        elif algo_info is not None and "be_deterministic" in algo_info:
            deterministic = algo_info["be_deterministic"]
        else:
            deterministic = False

        # switches noise off or on if Tianshou policy is DDPG
        if deterministic:
            self._ts_policy.eval()
        else:
            self._ts_policy.train()

        with torch.no_grad():
            # if not deterministic, return fully random action in a fraction of all queries
            if not deterministic and self._fully_random_fraction > 0. and \
                    np.random.rand() < self._fully_random_fraction:
                action = self.action_space.sample()
            else:
                policy_input_obs = self.observation_space.flatten_value(observation)
                batch = Batch(obs = policy_input_obs[None], info={})
                if deterministic:
                    net_output, _ = self._hidden_net(batch.obs)
                    if isinstance(self._hidden_net, SACNetwork):
                        mu, sigma = net_output
                        # map from R^n to action space
                        action = torch.tanh(mu[0])
                    else:
                        action = net_output[0]
                    action = action.detach().numpy()
                else:
                    action = self._ts_policy.forward(batch).act[0].detach().numpy()

                # map action from [-1, 1]^n to flat action space 
                # (NOTE: This has no effect on entropy in SAC)
                action = action*self._action_scale + self._action_bias

                # map flat action back to original action space
                action = self.action_space.unflatten_value(action)

        return 0, action

    def entropy(observation):
        policy_input_obs = self.observation_space.flatten_value(observation)
        batch = Batch(obs = policy_input_obs[None], info={})
        net_output, _ = self._hidden_net(batch.obs)
        mu, sigma = net_output
        return torch.log(sigma*torch.sqrt(2.*torch.pi)).sum() + 0.5*sigma.size()


