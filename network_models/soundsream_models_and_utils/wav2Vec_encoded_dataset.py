import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

inputcolumn = "encoded"
labelcolumn = "emotionCode"
dataset_column = "dataset"
clear_label_colums = "clear_emotion"


class W2V_EncDs(Dataset):

    def __init__(self, emo_dataset, device, model, encode = True,  data_colator = None, preencoded_dataset = None, to_oh =False):

        self.data_colator = data_colator
        self.to_oh = to_oh
        self.preencoded_dataset = preencoded_dataset
        self.model = model
        self.device = device
        self.emo_dataset = emo_dataset
        self.num_labels = len(emo_dataset.label_list)
        if encode:
            self.df = self.encodewithWav2Vec()


    def __getitem__(self, idx):
        if self.to_oh:
            return self.df[inputcolumn].iloc[idx], self.indices_to_one_hot(self.df[labelcolumn].iloc[idx])[0]
        else:
            return self.df[inputcolumn].iloc[idx], self.df[labelcolumn].iloc[idx]



    def indices_to_one_hot(self, data):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(self.num_labels)[targets]


    def __len__(self):
        return len(self.df[inputcolumn])


    def saveToPkl(self, path, filename):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(path+filename)

    def fromPkl(self, path, filename):
        self.df = pd.read_pickle(path+filename)

    def encodewithWav2Vec(self):


        dl = DataLoader(self.preencoded_dataset, shuffle=True, batch_size=6, num_workers=2 ,collate_fn= self.data_colator.collate_fn)

        def emoToId(emotion: str):
            return np.where(self.emo_dataset.label_list == emotion)[0]



        tensors = []
        emotions = []
        #emotions_names = []
        dataset_names = []
        print("assi")
        i = 0
        for batch, (X, z) in enumerate(dl):
            i = i+1
            from utils.Visual_Coding_utils import progress
            progress(i, self.preencoded_dataset.__len__()//6, "generating final dataset wav 2 vec")
            with torch.no_grad():
                data = self.model(X.to(self.device))
            tensors.extend(data.detach().cpu())
            emotions.extend(z)


        #for sample in iter(self.preencoded_dataset):
        #    i += 1
        #    from utils.Visual_Coding_utils import progress
        #    progress(i, self.preencoded_dataset.__len__(), "generating final dataset wav 2 vec")
        #    with torch.no_grad():
        #        t = torch.tensor(sample[0])
        #        t = torch.unsqueeze(t, dim=0).to(self.device)
        #        data = self.model(t)
#
        #    tensors.append(data[0].detach().cpu())
        #    emotions.append(sample[1])
            #emotions_names.append(sample[1])

        df = pd.DataFrame()
        #print(tensors)
        df[inputcolumn] = tensors
        df[labelcolumn] = emotions

        return df

        itersize = 2500
        df = pd.DataFrame()
        dfInter = pd.DataFrame()

        parts = len(tensors) // itersize
        end = len(tensors) % itersize

        df[inputcolumn] = tensors[0:itersize]
        df[labelcolumn] = emotions[0:itersize]
        #df[clear_label_colums] = emotions_names[0:itersize]
        for i in range(parts - 1):
            dfInter = pd.DataFrame()
            dfInter[inputcolumn] = tensors[(i + 1) * itersize:(i + 2) * itersize]
            dfInter[labelcolumn] = emotions[(i + 1) * itersize:(i + 2) * itersize]
            #dfInter[clear_label_colums] = emotions_names[(i + 1) * itersize:(i + 2) * itersize]
            gc.collect()
            df = df.append(dfInter)

        dfInter = pd.DataFrame()
        gc.collect()
        if itersize < len(tensors) and end != 0:
            dfInter[inputcolumn] = tensors[(parts * itersize):(parts * itersize + end)]
            dfInter[labelcolumn] = emotions[(parts * itersize):(parts * itersize + end)]
            #dfInter[clear_label_colums] = emotions_names[(parts * itersize):(parts * itersize + end)]
            gc.collect()
            df = df.append(dfInter)
        gc.collect()

        # df[self.labelcolumn] = emotions
        return df
