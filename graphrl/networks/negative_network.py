import torch
from torch import nn
import torch.nn.functional as F

class NegativeNetwork(nn.Module):
    """Restricts output to negative components."""
    def __init__(self, core_module, device = "cpu"):
        """
        Args: 
            core_module: Module that is used to process inputs before sending through -exp.
        """
        super().__init__()

        self.core_module = core_module
        self.device = device
        self.nonlin = nn.LogSigmoid()


    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device = self.device, dtype = torch.float)
        return self.nonlin(self.core_module(x))
