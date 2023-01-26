from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import Lambda
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Config
from transformers.utils import ModelOutput

@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class W2V_EmotionClassifierSevenEmos(nn.Module):

    def __init__(self):
        super(W2V_EmotionClassifierSevenEmos, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=768, out_features=300)
        self.activation = torch.nn.LeakyReLU()
        self.activationT = nn.Tanh()
        self.dropouts = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(300, 300)
        self.linear3 = torch.nn.Linear(300, 100)
        self.linear4 = torch.nn.Linear(100, 4)
        self.linear5 = torch.nn.Linear(4, 7)
        #pass dimension (along axis 0)
        self.softmax = torch.nn.Softmax(dim=0)

        self.initialize_weights()

    def forward(self, x, labels=None):
        x = self.linear1(x)
        x = self.activationT(x)
        x = self.dropouts(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        y = self.activationT(x)
        y = self.linear5(y)
        #y = self.softmax(y)

        return y


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

                if m.bias is not None:
                    nn.init.constant_(m.bias,0)



class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, model_name_or_path, pooling_mode, ):
        config = Wav2Vec2Config(name_or_path=model_name_or_path)
        super().__init__(config)
        self.pooling_mode = pooling_mode


        self.wav2vec2 = Wav2Vec2Model(config).to("cuda")
        self.classifier = W2V_EmotionClassifierSevenEmos().to("cuda")

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            labels,
            attention_mask
    ):
        outputs = self.wav2vec2(
            input_values,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        self.config.problem_type = "multi_label_classification"

        #ohe = Lambda(lambda y: torch.zeros(7, dtype=torch.float).to("cuda").scatter_(dim=1, index=torch.tensor(y), value=1))
        return logits
        # loss_fct = BCEWithLogitsLoss()
        # loss = loss_fct(logits, labels)

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return ClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs


