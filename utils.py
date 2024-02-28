import torch
import torch.nn as nn

class Trigger(nn.Module):

    def __init__(self, size: int = 32, transparency: float = 1.) -> None:
        super().__init__()

        self.size = size
        self.mask = nn.Parameter(torch.rand(size, size,device=torch.device('cuda')),requires_grad=True)
        self.transparency = transparency
        self.trigger = nn.Parameter(torch.rand(3, size, size,device=torch.device('cuda')) * 4 - 2,requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transparency * self.mask * self.trigger + (1 - self.mask * self.transparency) * x
    
class UAP(nn.Module):

    def __init__(self, size: int = 32) -> None:
        super().__init__()

        self.size = size
        self.perturbation = nn.Parameter(torch.zeros(3, size, size,device=torch.device('cuda')),requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.perturbation
