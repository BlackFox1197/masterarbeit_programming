import torch
from torch import nn

from network_models.soundsream_models_and_utils.ss_base_model import SSBaseModel

class SS_Enc_Class_Dims(nn.Module):
    def __init__(self, dropout = 0.2, input_size=4, num_emotions=7, eval_mode = False):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, num_emotions)
        self.linear2 = torch.nn.Linear(num_emotions, num_emotions)
        self.dropouts = torch.nn.Dropout(dropout if not eval_mode else 0)

    def forward(self, x, return_with_dims=False, soft_max = False, eval_mode = False):
        tanh = torch.tanh
        x = self.linear1(x)
        x = tanh(x)
        x = self.linear2(x)

        return x
