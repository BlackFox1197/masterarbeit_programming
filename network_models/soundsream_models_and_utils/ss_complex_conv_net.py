import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from network_models.soundsream_models_and_utils.ss_base_model import SSBaseModel

kernelSize = 5


class SSComplexConvModel3Sec(SSBaseModel):

    def __init__(self, num_emotions=7, eval_mode = False):

        super().__init__(num_emotions=num_emotions)

        self.conv1 = nn.Conv2d(1, 6, kernelSize, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernelSize, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernelSize, stride=1)
        self.conv4 = nn.Conv2d(32, 64, kernelSize, stride=1)

        self.linear1 = torch.nn.Linear(12544, 300)
        self.linear2 = torch.nn.Linear(300, 300)

        self.dropouts = torch.nn.Dropout(0.2 if not eval_mode else 0)



    def forward(self, x, return_with_dims=False, soft_max = False, eval_mode = False):
        relu = F.relu
        tanh = F.tanh
        softmax = F.softmax
        pool2d = F.max_pool2d

        x = pool2d(relu(self.conv1(x)), (2, 2))
        x = self.dropouts(x) if not eval_mode else x
        x = pool2d(relu(self.conv2(x)), (2, 2))
        x = pool2d(relu(self.conv3(x)), (2, 2))
        x = pool2d(relu(self.conv4(x)), (2, 2))
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = relu(x)
        x = self.dropouts(x) if not eval_mode else x
        x = self.linear2(x)
        x = relu(x)
        return super().forward(x, return_with_dims=return_with_dims, soft_max=soft_max)
