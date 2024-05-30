from typing import Literal
import torch
from torch import nn
import torchvision


class Resnet(nn.Module):
    def __init__(
        self,
        block_nums: Literal[18, 34, 50] = 50,
        class_nums: int = 1000,
    ) -> None:

        super().__init__()

        if block_nums == 18:
            resnet_module = torchvision.models.resnet18
        elif block_nums == 34:
            resnet_module = torchvision.models.resnet34
        else:
            resnet_module = torchvision.models.resnet50
        
        self.model = resnet_module(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, class_nums)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        return self.model(x)


if __name__ == "__main__":
    _ = Resnet()
