import os
import sys

import numpy as np
import pandas as pd
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

    def __len__(self):
        label_list = self.dataFrame.groupby(self.labelcolumn)[self.labelcolumn].count().index.array.to_numpy()
        label_list = np.sort(label_list)
        return self.dataFrame.__len__(), label_list.__len__()

    def __getitem__(self, idx):
        audio = torchaudio.load(self.dataFrame.iloc[idx][self.inputcolumn])[0].to("cuda")
        print(audio)
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

    def __init__(self, dataSet: AudioEmotionTessDataset, soundStream: SoundStream):
        self.soundStream = soundStream
        self.dataSet = dataSet

    def __getitem__(self, index):
        return encodedData[index]


    def _encodeWithSoundStream(self):
        for sample in self.dataSet.dataFrame:
            self.soundStream.encoder()


    def progress(self, count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()