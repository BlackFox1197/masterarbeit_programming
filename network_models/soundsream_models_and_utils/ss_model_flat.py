import torch
from torch import nn
import torch.nn.functional as F


class SSFlatModel(nn.Module):

    def __init__(self, x_size=512, y_size=175):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__()

        self.linear1 = torch.nn.Linear(x_size * y_size, 8000)
        self.linear2 = torch.nn.Linear(8000, 300)
        self.linear3 = torch.nn.Linear(300, 100)
        self.linear4 = torch.nn.Linear(100, 4)
        self.linear5 = torch.nn.Linear(4, 7)
        self.dropouts = torch.nn.Dropout(0.2)

    def forward(self, x, return_with_dims=False):
        relu = F.relu
        tanh = F.tanh
        softmax = F.softmax

        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = relu(x)
        x = self.dropouts(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = tanh(x)
        x = self.linear4(x)
        y = tanh(x)
        y = self.linear5(y)
        y = softmax(y)

        if return_with_dims:
            return x, y

        return y