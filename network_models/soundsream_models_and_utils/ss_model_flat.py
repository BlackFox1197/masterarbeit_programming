import torch
from torch import nn
import torch.nn.functional as F

from network_models.soundsream_models_and_utils.ss_base_model import SSBaseModel


class SSFlatModel(SSBaseModel):

    def __init__(self, x_size=512, y_size=175, num_emotions=7):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__(num_emotions=num_emotions)

        self.linear1 = torch.nn.Linear(x_size * y_size, 3000)
        self.linear2 = torch.nn.Linear(3000, 300)


    def forward(self, x, return_with_dims=False, soft_max = False):
        relu = F.relu
        tanh = F.tanh
        softmax = F.softmax

        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = relu(x)
        x = self.dropouts(x)
        x = self.linear2(x)
        x = relu(x)
        return super().forward(x, return_with_dims=return_with_dims, soft_max=soft_max)