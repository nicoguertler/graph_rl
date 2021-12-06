import torch
from torch import nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """Multi layer perceptron."""
    def __init__(self, input_dim, units_per_layer, activation_fns, device = "cpu"):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super(MLPNetwork, self).__init__()

        self.device = device

        # if only one activation function is given, apply it to all layers
        if type(activation_fns) is not list:
            activation_fns = [activation_fns]*len(units_per_layer)

        assert len(units_per_layer) == len(activation_fns)

        self.layers = []
        in_dim = input_dim
        for n_units, act in zip(units_per_layer, activation_fns):
            self.layers.append(nn.Linear(in_dim, n_units))
            if act is not None:
                self.layers.append(act)
            in_dim = n_units
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device = self.device, dtype = torch.float)
        batch_size = x.shape[0]
        res = x.view(batch_size, -1)
        res = self.model(res)
        return res
