import torch
from torch import nn
import torch.nn.functional as F

from network_models.soundsream_models_and_utils.ss_base_model import SSBaseModel


class SSDimRedModel(SSBaseModel):

    def __init__(self, x_size=512, reduceDims = True, num_emotions=7):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__(num_emotions=num_emotions)

        self.reduceDims = reduceDims
        self.linear1 = torch.nn.Linear(x_size, 8000)
        self.linear2 = torch.nn.Linear(8000, 300)
        self.dropouts = torch.nn.Dropout(0.2)

    def forward(self, x, return_with_dims=False, soft_max = False):
        relu = F.relu
        tanh = F.tanh
        softmax = F.softmax

        if(self.reduceDims):
            x = torch.mean(x, dim=3)

        x = self.linear1(x)
        x = relu(x)
        x = self.dropouts(x)
        x = self.linear2(x)
        x = relu(x)
        return super().forward(x, return_with_dims=return_with_dims, soft_max=soft_max)