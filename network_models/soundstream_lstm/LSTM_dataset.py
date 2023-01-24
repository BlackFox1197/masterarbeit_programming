import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from audiolm_pytorch import SoundStream
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.io import read_image
from torchvision.transforms import Lambda
from transformers import Wav2Vec2Processor


# class NewAudioEmotionTessDataset(Dataset):
#     inputcolumn = "path"
#     labelcolumn = "emotion"
#
#     def __init__(self, directory, soundstream: SoundStream):
#         self.directory = directory
#         self.dataFrame = self.load_custom_dataset()
#         label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
#         self.label_list = np.sort(label_list)
#         self.target_transform = Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
#         self.soundstream = soundstream
#
#
#     def __len__(self):
#         return self.dataFrame.__len__()
#
#
#     def emoToId(self, emotion: str):
#         return np.where(self.label_list == emotion)[0]
#
#     def getEmotionFromId(self, id: int):
#         return self.label_list[id]
#
#     def __getitem__(self, idx):
#         audio = self.soundstream.encoder(torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to("cuda"))
#         audio = torch.nn.functional.pad(audio, (0, 400 - audio.shape[1], 0, 0)).detach()
#         label = self.target_transform(self.emoToId(self.dataFrame.iloc[idx][self.labelcolumn]))
#         print("oh no")
#         return audio, label
#
#     def load_custom_dataset(self) -> pd.DataFrame:
#         paths = []
#         emotions = []
#         for dirname, _, filenames in os.walk(self.directory):
#             for filename in filenames:
#                 label = filename.split('_')[-1]
#                 label = label.split('.')[0]
#                 emotions.append(label.lower())
#                 paths.append(os.path.join(dirname, filename))
#         df = pd.DataFrame()
#         df[self.inputcolumn] = paths
#         df[self.labelcolumn] = emotions
#         return df


class AudioEmotionTessDataset(Dataset):
    inputcolumn = "path"
    labelcolumn = "emotion"

    def __init__(self, directory, device="cuda"):
        self.device = device
        self.directory = directory
        self.dataFrame = self.load_custom_dataset()
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        self.label_list = np.sort(label_list)
        self.samplerate = torchaudio.load(self.dataFrame.iloc[0][self.inputcolumn])[1]
        self.target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


    def __len__(self):
        return self.dataFrame.__len__(), self.label_list.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to(self.device)
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





class AudioEmotionTessWav2VecDataset(Dataset):
    inputcolumn = "encoded"
    labelcolumn = "emotionCode"

    def __init__(self, dataSet: AudioEmotionTessDataset, processor: Wav2Vec2Processor, sampling_rate: int):
        self.processor = processor
        self.dataSet = dataSet
        self.sampling_rate = sampling_rate
        self.encodedData = self._encodeWithSoundStream()
        self.target_transform = Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    def __getitem__(self, index):
        return self.encodedData.iloc[index][self.inputcolumn], self.target_transform(self.encodedData.iloc[index][self.labelcolumn])


    def __len__(self):
        return len(self.encodedData)

    def emoToId(self, emotion: str):
        return np.where(self.dataSet.label_list == emotion)[0]

    def getEmotionFromId(self, id: int):
        return self.dataSet.label_list[id]

    def _encodeWithSoundStream(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = []
        emotions = []
        i = 0
        for sample in iter(self.dataSet):
            i += 1
            from utils.Visual_Coding_utils import progress
            progress(i, self.dataSet.__len__()[0], "generating encoding")
            #resampled = librosa.resample(sample[0].numpy(), orig_sr=self.dataSet.samplerate, target_sr=self.sampling_rate)
            data = self.processor(sample[0], sampling_rate = self.sampling_rate)
            unWrapedData = data['input_values'][0][0]
            if(unWrapedData.shape[0] < 60000):
                diff = 60000 - unWrapedData.shape[0]
                unWrapedData = np.pad(unWrapedData, (diff//2, (diff//2)+1), mode="constant", constant_values=np.pi*0.1)
            unWrapedData = unWrapedData[0:60000]
            tensors.append(unWrapedData)
            #tensors.append(torch.nn.functional.pad(data, (0, 400 - data.shape[1], 0, 0)).detach().cpu().numpy())
            emotions.append(self.emoToId(sample[1]))
            torch.cuda.empty_cache()
        df = pd.DataFrame()
        df[self.inputcolumn] = tensors
        df[self.labelcolumn] = emotions
        return df

class AudioEmotionTessSoundStreamEncodedDataset(Dataset):
    inputcolumn = "encoded"
    labelcolumn = "emotionCode"

    def __init__(self, dataSet: AudioEmotionTessDataset, soundStream: SoundStream):
        self.soundStream = soundStream
        self.dataSet = dataSet
        self.encodedData = self._encodeWithSoundStream()
        self.target_transform = Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    def __getitem__(self, index):
        return self.encodedData.iloc[index][self.inputcolumn], self.target_transform(self.encodedData.iloc[index][self.labelcolumn])


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
            tensors.append(torch.nn.functional.pad(data, (0, 400 - data.shape[1], 0, 0)).detach().cpu().numpy())
            emotions.append(self.emoToId(sample[1]))
            torch.cuda.empty_cache()
        df = pd.DataFrame()
        df[self.inputcolumn] = tensors
        df[self.labelcolumn] = emotions
        return df

