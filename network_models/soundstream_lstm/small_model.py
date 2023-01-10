from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.utils import ModelOutput


class EmotionClassifierSevenEmos(nn.Module):

    def __init__(self):
        super(EmotionClassifierSevenEmos, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=512*400, out_features=200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 4)
        self.linear4 = torch.nn.Linear(4, 7)
        #pass dimension (along axis 0)
        self.softmax = torch.nn.Softmax(dim=0)

        self.initialize_weights()

    def forward(self, x, labels=None):
        x = torch.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        y = self.activation(x)
        y = self.linear4(y)
        y = self.softmax(y)

        return y

        #logits = y[0]
        #loss_fct = BCEWithLogitsLoss()
        #loss = loss_fct(logits, labels[0])
        # hidden_states = y[0]
        # hidden_states = self.merged_strategy(hidden_states)
        # loss_fct = BCEWithLogitsLoss()
        # loss = loss_fct(hidden_states, labels[0])
        # print("loss")
        # print(loss)
        # print("hidden states")
        # print(hidden_states)
        # return ClassifierOutput(
        #     loss=loss,
        #     logits=hidden_states,
        # )

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=0)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)




@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
