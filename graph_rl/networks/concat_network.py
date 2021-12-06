import torch
from torch import nn
import torch.nn.functional as F

class ConcatNetwork(nn.Module):
    """Concatenates all inputs."""
    def __init__(self, core_module, device = "cpu"):
        """
        Args: 
            core_module: Module that is used to process inputs once they have been concatenated.
        """
        super(ConcatNetwork, self).__init__()

        self.core_module = core_module
        self.device = device


    def forward(self, *args):
        inputs = []
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                inputs.append(torch.tensor(arg, device = self.device, dtype = torch.float))
            else:
                inputs.append(arg)
        return self.core_module(torch.cat(inputs, dim = 1))
