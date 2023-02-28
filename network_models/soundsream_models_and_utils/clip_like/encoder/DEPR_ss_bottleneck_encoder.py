import torch
from torch import nn


class SSBottelneckLayer(nn.Module):

    def __init__(self, num_cols, bottleneck_size, dropout=0.2, onlyDims = False, train_mode = True):
        self.onlyDims = onlyDims
        super().__init__()

        self.linear1 = torch.nn.Linear(512 * num_cols, 512 * num_cols // 100)
        self.linear2 = torch.nn.Linear(512 * num_cols // 100, 300)
        self.linear3 = torch.nn.Linear(300, 30)
        self.linear4 = torch.nn.Linear(30, bottleneck_size)
        self.code = torch.nn.Linear(bottleneck_size, bottleneck_size)
        self.linear4m = torch.nn.Linear(bottleneck_size, 30)
        self.linear3m = torch.nn.Linear(30, 300)
        self.linear2m = torch.nn.Linear(300, 512 * num_cols // 100)
        self.linear1m = torch.nn.Linear(512 * num_cols // 100, 512 * num_cols)

        self.droput = torch.nn.Dropout(dropout)


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
        x = tanh(x)
        y = self.linear4m(x)
        y = tanh(y)
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
