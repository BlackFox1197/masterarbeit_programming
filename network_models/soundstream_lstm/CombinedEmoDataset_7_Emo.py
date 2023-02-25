import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from math import floor
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Lambda


def crawl(labelLambda: callable, directory):
    paths = []
    emotions = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            label = labelLambda(filename)
            if (label == -1):
                continue
            emotions.append(label.lower())
            paths.append(os.path.join(dirname, filename))

    return paths, emotions


def loadMesd(directory_mesd) -> Tuple:
    mesdDict = {
        'Anger': "angry",
        'Disgust': "disgust",
        'Happiness': "happy",
        'Neutral': "neutral",
        'Fear': "fear",
        'Sadness': "sad"
    }



    def lamb(filename):
        label = filename.split('_')[0]
        label = mesdDict[label]
        return label

    paths, emotions = crawl(lamb, directory_mesd)

    return paths, emotions


def loadCafe(directory_cafe) -> Tuple:
    cafeDict = {
        'C': "angry",
        'D': "disgust",
        'J': "happy",
        'N': "neutral",
        'P': "fear",
        'S': "surprise",
        'T': "sad"
    }

    def lamb(filename):
        label = filename.split('-')[1]
        label = label.split('-')[0]
        label = cafeDict[label]
        return label

    paths, emotions = crawl(lamb, directory_cafe)
    return paths, emotions


def loadInduced(directory_induced) -> Tuple:
    induced_dict = {
        'angry': "angry",
        'disgusted': "disgust",
        'happy': "happy",
        'neutral': "neutral",
        'fearful': "fear",
        'surprised': "surprise",
        'sad': "sad"
    }

    def lamb(filename):
        label = filename.split('_')[0]
        label = induced_dict[label]
        return label

    paths, emotions = crawl(lamb, directory_induced)
    return paths, emotions

# https://zenodo.org/record/1188976
def loadRavdess(directory_ravdess) -> Tuple:
    ravedessDict = {
        "01": "neutral",
        # "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise"
    }

    def lamb(filename):
        label = filename.split('-')[2]
        if (label == "02"):
            label = -1
        else:
            label = ravedessDict[label]
        return label

    paths, emotions = crawl(lamb, directory_ravdess)

    return paths, emotions


def loadTess(directory_tess):
    def lamb(filename):
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        if (label == "ps"):
            label = "surprise"
        return label

    paths, emotions = crawl(lamb, directory_tess)
    return paths, emotions


def collateToSeconds(seconds, samplingRate, const_value=0, asCallable=True, data=[], circular_padding = False, device = "cuda"):
    def collator(dataInner):
        tarLength = floor(seconds * samplingRate)
        currLen = len(dataInner)
        parr = currLen % 2
        func: callable

        if (currLen < tarLength):
            paddboth = int((tarLength - currLen) / 2)
            if(circular_padding and False):
                return torch.tensor(np.pad(dataInner.cpu(), (paddboth, paddboth + parr), mode="wrap")).to(device)
            return nn.functional.pad(dataInner, (paddboth, paddboth + parr), value=const_value)
        else:
            cut = int((currLen - tarLength) / 2)
            return dataInner[cut:(tarLength + cut)]

    return collator if asCallable else collator(data)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


class CombinedEmoDataSet_7_emos(Dataset):
    inputcolumn = "path"
    labelcolumn = "emotion"

    def     __init__(self, directory_tess: None | str = None, directory_ravdess: None | str = None,
                 directory_cafe: None | str = None, directory_mesd: None | str = None, directory_induced: None | str = None,
                 transFormAudio: callable = lambda x: x, device="cuda", filter_emotions: None | List[str] = None):
        self.device = device

        self.directory_induced = directory_induced
        self.directory_tess = directory_tess
        self.directory_mesd = directory_mesd
        self.directory_ravdess = directory_ravdess
        self.directory_cafe = directory_cafe

        self.filter_emotions = filter_emotions
        self.transFormAudio = transFormAudio
        self.dataFrame = self.load_custom_dataset()
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        self.label_list = np.sort(label_list)
        # self.target_transform = Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to(self.device)
        #audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0]
        audio = torch.unsqueeze(self.transFormAudio(audio[0]), dim=0)
        label = self.dataFrame.iloc[idx][self.labelcolumn]
        return audio, label


    def load_custom_dataset(self) -> pd.DataFrame:
        tess_paths, tess_emo = self.loadTess() if self.directory_tess is not None else ([], [])
        incuced_paths, incuced_emo = self.loadInduced() if self.directory_induced is not None else ([], [])
        mesd_paths, mesd_emo = self.loadMesd() if self.directory_mesd is not None else ([], [])
        ravdess_paths, ravdess_emo = self.loadRavdess() if self.directory_ravdess is not None else ([], [])
        cafe_paths, cafe_emo = self.loadCafe() if self.directory_cafe is not None else ([], [])

        paths = mesd_paths + tess_paths + cafe_paths + ravdess_paths + incuced_paths
        emotions = mesd_emo + tess_emo + cafe_emo + ravdess_emo + incuced_emo

        df = pd.DataFrame()
        df[self.inputcolumn] = paths
        df[self.labelcolumn] = emotions

        if self.filter_emotions is not None:
            df = df[~df[self.labelcolumn].isin(self.filter_emotions)]

        return df

    def loadMesd(self) -> Tuple:
        return loadMesd(self.directory_mesd)

    def loadInduced(self) -> Tuple:
        return loadInduced(self.directory_induced)

    def loadCafe(self) -> Tuple:
        return loadCafe(self.directory_cafe)

    # https://zenodo.org/record/1188976
    def loadRavdess(self) -> Tuple:
        return loadRavdess(self.directory_ravdess)

    def loadTess(self):
        return loadTess(self.directory_tess)


class DatasetGeneric(Dataset):
    inputcolumn = "path"
    labelcolumn = "emotion"

    def __init__(self, directory, load_dataset: callable, device="cuda", transFormAudio: callable = lambda x: x,):
        self.loadDataset: callable = load_dataset
        self.device = device
        self.directory = directory
        self.dataFrame = self.load_custom_dataset()
        self.transFormAudio = transFormAudio
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        self.label_list = np.sort(label_list)
        self.target_transform = Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to(self.device)
        audio = torch.unsqueeze(self.transFormAudio(audio[0]), dim=0)
        label = self.dataFrame.iloc[idx][self.labelcolumn]
        return audio, label

    def getDir(self, idx):
        label = self.dataFrame.iloc[idx][self.labelcolumn]
        return self.dataFrame.iloc[idx][self.inputcolumn], label
    def load_custom_dataset(self) -> pd.DataFrame:
        paths, emotions = self.loadDataset(self.directory)

        df = pd.DataFrame()
        df[self.inputcolumn] = paths
        df[self.labelcolumn] = emotions
        return df
