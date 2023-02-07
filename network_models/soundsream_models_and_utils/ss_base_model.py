import torch
import torch.nn.functional as F
from torch import nn


class SSBaseModel(nn.Module):
    def __init__(self, num_emotions=7):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__()

        self.base_linear1 = torch.nn.Linear(300, 100)
        self.base_linear2 = torch.nn.Linear(100, 4)
        self.base_linear3 = torch.nn.Linear(4, 4)
        self.base_linear4 = torch.nn.Linear(4, num_emotions)
        self.dropouts = torch.nn.Dropout(0.2)

    def forward(self, x, return_with_dims=False, soft_max=False):
        relu = F.relu
        tanh = F.tanh
        softmax = F.softmax

        x = self.base_linear1(x)
        x = tanh(x)
        x = self.base_linear2(x)
        y = tanh(x)
        y = self.base_linear3(y)
        y = tanh(y)
        y = self.base_linear4(y)

        y = softmax(y) if soft_max else y

        if return_with_dims:
            return x, y

        return y