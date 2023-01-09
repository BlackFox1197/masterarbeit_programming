import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from audiolm_pytorch import SoundStream
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.io import read_image

class AudioEmotionTessDataset(Dataset):
    inputcolumn = "path"
    labelcolumn = "emotion"

    def __init__(self, directory):
        self.directory = directory
        self.dataFrame = self.load_custom_dataset()
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        self.label_list = np.sort(label_list)


    def __len__(self):
        return self.dataFrame.__len__(), self.label_list.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to("cuda")
        label = self.dataFrame.iloc[idx][self.labelcolumn]
        return audio, label

    def load_custom_dataset(self) -> pd.DataFrame:
        paths = []
        emotions = []
        for dirname, _, filenames in os.walk(self.directory):
            for filename in filenames:
                label = filename.split('_')[-1]
                label = label.split('.')[0]
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))
        df = pd.DataFrame()
        df[self.inputcolumn] = paths
        df[self.labelcolumn] = emotions
        return df





class AudioEmotionTessSoundStreamEncodedDataset(Dataset):
    inputcolumn = "encoded"
    labelcolumn = "emotionCode"

    def __init__(self, dataSet: AudioEmotionTessDataset, soundStream: SoundStream):
        self.soundStream = soundStream
        self.dataSet = dataSet
        self.encodedData = self._encodeWithSoundStream()

    def __getitem__(self, index):
        return self.encodedData.iloc[index][self.inputcolumn], self.encodedData.iloc[index][self.labelcolumn]


    def __len__(self):
        return len(self.encodedData)

    def emoToId(self, emotion: str):
        return np.where(self.dataSet.label_list == emotion)[0]

    def getEmotionFromId(self, id: int):
        return self.dataSet.label_list[id]

    def _encodeWithSoundStream(self):
        tensors = []
        emotions = []
        i = 0
        for sample in iter(self.dataSet):
            i += 1
            from utils.Visual_Coding_utils import progress
            progress(i, self.dataSet.__len__()[0], "generating encoding")
            data = self.soundStream.encoder(sample[0])
            tensors.append(torch.nn.functional.pad(torch.tensor(data).to("cuda"), (0, 400 - data.shape[1], 0, 0)).detach().cpu().numpy())
            emotions.append(self.emoToId(sample[1]))
            torch.cuda.empty_cache()
        df = pd.DataFrame()
        df[self.inputcolumn] = tensors
        df[self.labelcolumn] = emotions
        return df

