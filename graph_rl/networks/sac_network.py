import torch
from torch import nn

class SACNetwork(nn.Module):
    """Outputs mean, variance and state (None) as required by Tianshou implementation of SAC."""
    def __init__(self, input_processing_fn, n_hidden, output_dim, device, 
            clip_range_sigma=None, clip_range_mu=None):
        """
        Args: 
            input_processing_fn: pytorch module which outputs a tensor with shape (batch_size, n_hidden).
        """
        super().__init__()
        self.device = device
        self.input_processing_fn = input_processing_fn
        self.mu_fn = nn.Linear(n_hidden, output_dim)
        if clip_range_sigma is None:
            self.clip_range_sigma = [1.0e-3, 6.0e0]
        else:
            self.clip_range_sigma = clip_range_sigma
        if clip_range_mu is None:
            # Due to squashing this is  close to (-1, 1)
            self.clip_range_mu = [-3., 3.]
        else:
            self.clip_range_mu = clip_range_mu
        self.sigma_fn = nn.Linear(n_hidden, output_dim)

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device = self.device, dtype = torch.float)
        batch_size = s.shape[0]
        s_flat = s.view(batch_size, -1)
        hidden_output = self.input_processing_fn(s_flat)
        mu = self.mu_fn(hidden_output)
        sigma = self.sigma_fn(hidden_output).exp()
        sigma = torch.clamp(sigma, min=self.clip_range_sigma[0], 
            max=self.clip_range_sigma[1])
        mu = torch.clamp(mu, min=self.clip_range_mu[0], 
            max=self.clip_range_mu[1])
        return (mu, sigma), None
