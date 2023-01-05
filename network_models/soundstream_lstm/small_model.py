import torch
from torch import nn


class EmotionClassifierSevenEmos(nn.Module):

    def __init__(self):
        super(EmotionClassifierSevenEmos, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=102400, out_features=200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 4)
        self.linear4 = torch.nn.Linear(4, 6)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        y = self.activation(x)
        y = self.linear4(y)
        y = self.softmax(y)
        return y, x