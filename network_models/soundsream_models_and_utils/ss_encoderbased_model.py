import torch

from network_models.soundsream_models_and_utils.ss_base_model import SSBaseModel

class SSClipBasedModel(SSBaseModel):
    def __init__(self, dropout = 0.2, input_size=1024, num_emotions=7, eval_mode = False):  # 175 is equivalent to 3,5 seconds with a sampling-rate of 16000
        super().__init__(num_emotions=num_emotions)

        self.linear1 = torch.nn.Linear(input_size, 300)
        self.dropouts = torch.nn.Dropout(dropout if not eval_mode else 0)

    def forward(self, x, return_with_dims=False, soft_max = False, eval_mode = False):
        relu = torch.relu

        x = self.linear1(x)
        x = relu(x)
        x = self.dropouts(x) if not eval_mode else x
        return super().forward(x, return_with_dims=return_with_dims, soft_max=soft_max)
