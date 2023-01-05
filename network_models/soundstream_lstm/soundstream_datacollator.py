from typing import Dict, List, Optional, Union, Any
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import Trainer, is_apex_available

if is_apex_available():
    from apex import amp


class SoundStreamDataCollator:
    inputcolumn = "encoded"
    labelcolumn = "emotionCode"



    def __init__(self, length):
        self.length = length




    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{self.inputcolumn: feature[self.inputcolumn]} for feature in features]
        label_features = [feature[self.labelcolumn] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float


        input_batch = [torch.nn.functional.pad(feature[1], (0, 200 - feature[1].shape[1], 0, 0)) for feature in input_features]
        batch = {self.inputcolumn: input_batch,
                 self.labelcolumn: torch.tensor(label_features, dtype=d_type)}
        # torch.tensor(encoded_dataset.__getitem__(3)[0])
        # tensor = torch.nn.functional.pad(tensor, (0, 200 - tensor.shape[1], 0, 0))
        # tensor = tensor.flatten()

        # batch = self.processor.pad(
        #     input_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )

        return batch




class SoundstreamModelTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.empty_cache()

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()