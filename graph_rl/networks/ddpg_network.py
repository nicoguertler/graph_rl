import torch
from torch import nn

class DDPGNetwork(nn.Module):
    """Outputs action and state (None) as required by Tianshou implementation of DDPG.
    
    The output of input_processing is squashed using tanh."""
    def __init__(self, input_processing, max_action, n_hidden, output_dim, device):
        """
        Args: 
            input_processing: pytorch module
        """
        super().__init__()
        self.device = device
        self.input_processing = input_processing
        self.max_action = torch.tensor(max_action)
        self.mu_fn = nn.Linear(n_hidden, output_dim)

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device = self.device, dtype = torch.float)
        batch_size = s.shape[0]
        s_flat = s.view(batch_size, -1)
        orig_output = self.input_processing(s_flat)
        mu = self.mu_fn(orig_output)
        action = self.max_action*torch.tanh(mu)
        return action, None
