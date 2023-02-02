from ctypes import Union
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: bool | str = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    num_labels: int = 7

    # def __init__(self, num_labels, **kwargs):
    #     super.__init__(**kwargs)
    #     self.num_labels = num_labels

    def __call__(self, features: List[Dict[str, List[int] | torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature[0]} for feature in features]
        label_features = [feature[1] for feature in features]

        #d_type = torch.long if isinstance(label_features[0], int) else torch.float
        #self.indices_to_one_hot(label_features)

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(self.indices_to_one_hot(label_features))

        print(batch)
        return batch


    """Convert an iterable of indices to one-hot encoded labels."""
    def indices_to_one_hot(self, data):

        targets = np.array(data).reshape(-1)
        return np.eye(self.num_labels)[targets]

    def collate_fn(self, batch):
        input_features = [{"input_values": feature[0]} for feature in batch]
        label_features = [feature[1] for feature in batch]
        batch_features = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            #pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        batch_label = torch.tensor(self.indices_to_one_hot(label_features))

        return batch_features["input_values"], batch_label


