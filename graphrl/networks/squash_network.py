import torch
from torch import nn
import torch.nn.functional as F

class SquashNetwork(nn.Module):
    """Squashes output to n-dim interval via tanh."""
    def __init__(self, core_module, low, high, device = "cpu"):
        """
        Args: 
            core_module: Module that is used to process inputs before squashing.
        """
        super(SquashNetwork, self).__init__()

        self.core_module = core_module
        self.device = device
        self._offset = torch.tensor(0.5*(high + low), device = self.device, dtype = torch.float)
        self._scale = torch.tensor(0.5*(high - low), device = self.device, dtype = torch.float)


    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device = self.device, dtype = torch.float)
        return self._scale*torch.tanh(self.core_module(x)) + self._offset
