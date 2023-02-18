import numpy as np
import torch
from torch import nn

kernelSize = 5

class SS_Direct_Downmapping_Conv_Model(nn.Module):


    def __init__(self, x_size, y_size, output= 1024, dropout=0.2, eval_mode=False):
        super().__init__()
        enda = int((((x_size - np.floor(kernelSize / 2) * 2) / 2) - (np.floor(kernelSize / 2) * 2)) // 2)
        endb = int((((y_size - np.floor(kernelSize / 2) * 2) / 2) - (np.floor(kernelSize / 2) * 2)) // 2)


        self.conv1 = nn.Conv2d(1, 6, kernelSize)
        self.conv2 = nn.Conv2d(6, 16, kernelSize)

        self.linear1 = torch.nn.Linear(16 * endb * enda, output)

        self.dropouts = torch.nn.Dropout(dropout if not eval_mode else 0)




    def forward(self, x, eval_mode = False):
        relu = torch.relu
        pool2d = torch.max_pool2d

        x = pool2d(relu(self.conv1(x)), (2, 2))
        x = self.dropouts(x) if not eval_mode else x
        x = pool2d(relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)

        x = self.linear1(x)

        return x / x.norm(dim=1, keepdim=True)