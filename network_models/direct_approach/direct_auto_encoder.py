import torch
from torch import nn


class DirectBottelneckEncoder(nn.Module):

    def __init__(self, size, bottleneck_size, dropout=0.2, onlyDims = False, train_mode = True):
        self.onlyDims = onlyDims
        super().__init__()

        self.linear1 = nn.Linear(size, size // 100)
        self.linear2 = nn.Linear(size // 100, 300)
        self.linear3 = nn.Linear(300, 30)
        self.linear4 = nn.Linear(30, bottleneck_size)
        self.code = nn.Linear(bottleneck_size, bottleneck_size)
        self.linear4m = nn.Linear(bottleneck_size, 30)
        self.linear3m = nn.Linear(30, 300)
        self.linear2m = nn.Linear(300, size// 100)
        self.linear1m = nn.Linear(size // 100, size)

        self.droput = nn.Dropout(dropout)


    def forward(self, x, return_with_dims = False, flatten = False):
        relu = torch.relu
        tanh = torch.tanh
        softmax = torch.softmax
        dropout = torch.dropout

        if flatten:
            x = x.flatten()
        x = self.linear1(x)
        x = relu(x)
        x = self.droput(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = relu(x)
        x = self.linear4(x)
        x = relu(x)
        x = self.code(x)
        x = relu(x)
        y = self.linear4m(x)
        y = relu(y)
        y = self.linear3m(y)
        y = relu(y)
        y = self.linear2m(y)
        y = relu(y)
        y = self.droput(y)
        y = relu(y)
        y = self.linear1m(y)

        if(self.onlyDims):
            return x

        if return_with_dims:
            return x, y

        return y