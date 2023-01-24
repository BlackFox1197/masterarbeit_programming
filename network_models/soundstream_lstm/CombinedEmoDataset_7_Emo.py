import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Lambda







class CombinedEmoDataSet_7_emos(Dataset):
    inputcolumn = "path"
    labelcolumn = "emotion"

    def __init__(self, directory_tess, directory_ravdess, directory_cafe, directory_mesd,  device="cuda"):
        self.device = device


        self.directory_tess = directory_tess
        self.directory_mesd = directory_mesd
        self.directory_ravdess = directory_ravdess
        self.directory_cafe = directory_cafe


        self.dataFrame = self.load_custom_dataset()
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        self.label_list = np.sort(label_list)
        self.samplerate = torchaudio.load(self.dataFrame.iloc[0][self.inputcolumn])[1]
        self.target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to(self.device)
        label = self.dataFrame.iloc[idx][self.labelcolumn]
        return audio, label

    def load_custom_dataset(self) -> pd.DataFrame:
        paths = []
        emotions = []

        mesd_paths, mesd_emo = self.loadMesd()
        tess_paths, tess_emo = self.loadTess()
        cafe_paths, cafe_emo = self.loadCafe()
        ravdess_paths, ravdess_emo = self.loadRavdess()

        paths = mesd_paths + tess_paths + cafe_paths + ravdess_paths
        emotions = mesd_emo + tess_emo + cafe_emo + ravdess_emo

        df = pd.DataFrame()
        df[self.inputcolumn] = paths
        df[self.labelcolumn] = emotions
        return df


    def loadMesd(self) -> Tuple:
        paths = []
        emotions = []
        mesdDict = {
            'Anger': "angry",
            'Disgust': "disgust",
            'Happiness': "happy",
            'Neutral': "neutral",
            'Fear': "fear  ",
            'Sadness': "sad"
        }
        for dirname, _, filenames in os.walk(self.directory_mesd):
            for filename in filenames:
                label = filename.split('_')[0]
                label = mesdDict[label]
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))

        return paths, emotions

    def loadCafe(self) -> Tuple:
        paths = []
        emotions = []
        cafeDict = {
            'C': "angry",
            'D': "disgust",
            'J': "happy",
            'N': "neutral",
            'P': "fear  ",
            'S': "surprise",
            'T': "sad"
        }
        for dirname, _, filenames in os.walk(self.directory_cafe):
            for filename in filenames:
                label = filename.split('-')[1]
                label = label.split('-')[0]
                label = cafeDict[label]
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))

        return paths, emotions

    # https://zenodo.org/record/1188976
    def loadRavdess(self) -> Tuple:
        paths = []
        emotions = []
        ravedessDict = {
            "01": "neutral",
            #"02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fear",
            "07": "disgust",
            "08": "surprise"
        }
        for dirname, _, filenames in os.walk(self.directory_ravdess):
            for filename in filenames:
                label = filename.split('-')[2]
                if(label == "02"):
                    continue
                label = ravedessDict[label]
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))

        return paths, emotions

    def loadTess(self):
        paths = []
        emotions = []
        for dirname, _, filenames in os.walk(self.directory_tess):
            for filename in filenames:
                label = filename.split('_')[-1]
                label = label.split('.')[0]
                if(label == "ps"):
                    label = "surprise"
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))

        return paths, emotions

    def crawl(self, labelLambda: callable):
        paths = []
        emotions = []
        for dirname, _, filenames in os.walk(self.directory_tess):
            for filename in filenames:
                label = labelLambda(filename)
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))

        return paths, emotions