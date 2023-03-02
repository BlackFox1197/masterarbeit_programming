import gc
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import umap
from audiolm_pytorch import SoundStream
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset

from network_models.clip.models.ss_encoder_downmapping import EncoderDownmapping
from network_models.soundsream_models_and_utils.clip_like.encoder.DEPR_ss_bottleneck_encoder import SSBottelneckLayer
from network_models.soundsream_models_and_utils.clip_like.mapping_down.ss_direct_downmapping import \
    SS_Direct_Downmapping_Model
from network_models.soundstream_lstm.CombinedEmoDataset_7_Emo import DatasetGeneric, CombinedEmoDataSet_7_emos, \
    collateToSeconds


class ss_encoded_dataset_full(Dataset):
    """ This dataset loads an audio dataset and encodes it with soundstream"""
    def __init__(self, sound_stream_path: None | str = None,
                 directory_tess: None | str = None, directory_ravdess: None | str = None,
                 directory_cafe: None | str = None, directory_mesd: None | str = None, directory_induced: None | str = None,
                 seconds = 3.5, sr=16000, device="cpu", one_hot_encoded = True, csvPath: str | None = None, clip_path: str | None = None, encoder =False, circular=False, umap = False, umap_dims = 4, just_mfcc = False):
        assert not (sound_stream_path is None and csvPath is None and not just_mfcc)
        super().__init__()
        self.device = device
        self.sr = sr
        self.seconds = seconds

        # lookup if soundstream path is available
        if sound_stream_path is not None and csvPath is None:
            soundstream = self.init_soundstream(sound_stream_path)
        else:
            soundstream = None

        # check for clip
        if clip_path is not None and sound_stream_path is not None and csvPath is None:
            clip = self.init_clip(clip_path, encoder)
        else:
            clip = None


        paths_dataset = CombinedEmoDataSet_7_emos(
            directory_tess=directory_tess, directory_cafe=directory_cafe, directory_ravdess=directory_ravdess, directory_induced = directory_induced,
            directory_mesd=directory_mesd, device=device, transFormAudio=collateToSeconds(seconds, sr, const_value=0, circular_padding=circular, device=device), with_dataset=True, librosa=just_mfcc)
        num_labels = len(paths_dataset.label_list)

        self.encoded_dataset = ss_encoded_dataset(data_set=paths_dataset, sound_stream=soundstream, device=device,
                                                  num_labels=num_labels, one_hot_encoded=one_hot_encoded, csvPath=csvPath, clip=clip, encoder=encoder, umap=umap, umap_dims= umap_dims, just_mfcc=just_mfcc)


    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def saveEncoding(self, path):
        self.encoded_dataset.saveEncoding(path)


    def getNextNeighbour(self, vector, emotion=None, included_datasets  = None):
        if included_datasets is None:
            included_datasets = [CombinedEmoDataSet_7_emos.cafeString, CombinedEmoDataSet_7_emos.tessString, CombinedEmoDataSet_7_emos.mesdString, CombinedEmoDataSet_7_emos.ravedessString]
        filterDatasets = self.encoded_dataset.encodedData.loc[self.encoded_dataset.encodedData[self.encoded_dataset.dataset_column].isin(included_datasets)]
        if emotion is not None:
            filterDatasets = filterDatasets.loc[filterDatasets[self.encoded_dataset.clear_label_colums].isin([emotion])]
        npds = np.asarray([fp.numpy() for fp in filterDatasets["pca"]])
        kdt = KDTree(npds, metric='euclidean')
        cv = kdt.query(np.atleast_2d(vector), k=1, return_distance=False)[0, 0]
        return filterDatasets.iloc[cv]

    def one_hot_to_id(self, x):
        pass

    def __len__(self):
        return len(self.encoded_dataset)

    def init_clip(self, clip_path, encoder =False) -> EncoderDownmapping:
        if(encoder):
            clip = SSBottelneckLayer(num_cols = 175,bottleneck_size =  4, dropout=0, train_mode=False, onlyDims=True).to(self.device)
        else:
            clip = EncoderDownmapping(embed_dim=512, n_heads=4, ff_dim=2, n_layers=1, dropout=0.2, output=1024, max_seq_len=int(self.seconds / (0.03 / (24000/self.sr)))).to(self.device)
        clip.load_state_dict(torch.load(clip_path))
        for param in clip.parameters():
            param.requires_grad = False
        return clip

    def init_soundstream(self, sound_stream_path) -> SoundStream:
        soundstream = SoundStream(
            codebook_size=1024,
            rq_num_quantizers=8,
            attn_window_size=256,  # local attention receptive field at bottleneck
            attn_depth=2
        ).to(self.device)
        soundstream.load(sound_stream_path)
        for param in soundstream.parameters():
            param.requires_grad = False
        return soundstream

class ss_encoded_dataset(Dataset):
    """ This dataset encodes an audio dataset with soundsream"""
    inputcolumn = "encoded"
    labelcolumn = "emotionCode"
    dataset_column = "dataset"
    pcacolumn = "pca"
    clear_label_colums = "clear_emotion"

    # save_encode_column = "encode_col"
    # save_dataset_column = "ds_col"

    def __init__(self, data_set: CombinedEmoDataSet_7_emos | DatasetGeneric,  clip: EncoderDownmapping | SS_Direct_Downmapping_Model | None,  sound_stream: SoundStream | None, num_labels = 7,
                 one_hot_encoded = True, device = "cpu", csvPath: str | None = None, encoder = False, umap = False, umap_dims = 4, just_mfcc = False):

        self.umap_dims = umap_dims
        self.umap = umap
        self.encoder = encoder
        self.clip = clip
        assert not (sound_stream is None and csvPath is None and not just_mfcc)
        super().__init__()
        self.device = device
        self.one_hot_encoded = one_hot_encoded
        self.soundStream = sound_stream
        self.num_labels = num_labels
        self.dataSet = data_set

        if just_mfcc:
            self.label_list = self.dataSet.label_list
            self.encodedData: pd.DataFrame = self.generateMfccs()
        else:
            if(csvPath is None):
                self.label_list = self.dataSet.label_list
                self.encodedData: pd.DataFrame = self._encodeWithSoundStream()
            else:
                self.encodedData = self.loadEncoding(csvPath)
                label_list = self.encodedData.groupby(self.clear_label_colums)[self.clear_label_colums].count().index.array.to_numpy()
                self.label_list = np.sort(label_list)

        self.target_transform = self.indices_to_one_hot

    def __getitem__(self, index):
        if(self.umap):
            return self.encodedData.iloc[index][self.pcacolumn], self.target_transform(self.encodedData.iloc[index][self.labelcolumn])[0]
        return self.encodedData.iloc[index][self.inputcolumn], self.target_transform(self.encodedData.iloc[index][self.labelcolumn])[0]


    def __len__(self):
        return len(self.encodedData)

    def indices_to_one_hot(self, data):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(self.num_labels)[targets]


    def saveEncoding(self, path):
        #Path(path).mkdir(parents=True, exist_ok=True)
        #saveDf[self.save_dataset_column] = self.dataSet

        self.encodedData.to_pickle(path)

    def loadEncoding(self, path):
        dataframe = pd.read_pickle(path)

        self.num_labels = max(dataframe[self.labelcolumn])[0] +1

        return dataframe

    def emoToId(self, emotion: str):
        return np.where(self.dataSet.label_list == emotion)[0]


    def getEmotionFromId(self, id: int):
        return self.dataSet.label_list[id]


    def generateMfccs(self):
        from utils.Visual_Coding_utils import progress
        tensors = []
        emotions = []
        emotions_names = []
        dataset_names = []
        i = 0
        for sample in iter(self.dataSet):
            i += 1
            progress(i, self.dataSet.__len__(), "generating encoding")
            mfcc = librosa.feature.mfcc(y=sample[0], sr=16000, n_mfcc=40).T
            dataset_names.append(sample[2])
            tensors.append(mfcc)
            emotions.append(self.emoToId(sample[1]))
            emotions_names.append(sample[1])

        df = pd.DataFrame()
        df[self.inputcolumn] = tensors
        df[self.dataset_column] = dataset_names
        df[self.labelcolumn] = emotions
        df[self.clear_label_colums] = emotions_names
        return df


    def _encodeWithSoundStream(self):
        tensors = []
        emotions = []
        emotions_names = []
        dataset_names = []
        i = 0
        for sample in iter(self.dataSet):
            i += 1
            from utils.Visual_Coding_utils import progress
            progress(i, self.dataSet.__len__(), "generating encoding")
            with torch.no_grad():
                data = self.soundStream(sample[0], return_encoded =True)

                if(self.clip is not None):
                    if(self.encoder):
                        data = self.clip(torch.flatten(torch.squeeze(data[0], dim=1), 1))
                    else:
                        data = self.clip(torch.transpose(torch.squeeze(data[0], dim=1), 1, 2))

            dataset_names.append(sample[2])
            tensors.append(data[0].detach().cpu())
            emotions.append(self.emoToId(sample[1]))
            emotions_names.append(sample[1])

        itersize = 2500
        df = pd.DataFrame()
        dfInter = pd.DataFrame()

        parts = len(tensors)//itersize
        end = len(tensors)%itersize

        df[self.inputcolumn] = tensors[0:itersize]
        df[self.dataset_column] = dataset_names[0:itersize]
        df[self.labelcolumn] = emotions[0:itersize]
        df[self.clear_label_colums] = emotions_names[0:itersize]
        for i in range(parts - 1):
            dfInter = pd.DataFrame()
            dfInter[self.inputcolumn] = tensors[(i+1)*itersize:(i+2)*itersize]
            dfInter[self.labelcolumn] = emotions[(i+1)*itersize:(i+2)*itersize]
            dfInter[self.dataset_column] = dataset_names[(i+1)*itersize:(i+2)*itersize]
            dfInter[self.clear_label_colums] = emotions_names[(i+1)*itersize:(i+2)*itersize]
            gc.collect()
            df = df.append(dfInter)

        dfInter = pd.DataFrame()
        gc.collect()
        if itersize < len(tensors) and end != 0:
            dfInter[self.inputcolumn] = tensors[(parts*itersize):(parts*itersize+end)]
            dfInter[self.dataset_column] = dataset_names[(parts*itersize):(parts*itersize+end)]
            dfInter[self.labelcolumn] = emotions[(parts*itersize):(parts*itersize+end)]
            dfInter[self.clear_label_colums] = emotions_names[(parts*itersize):(parts*itersize+end)]
            gc.collect()
            df = df.append(dfInter)
        gc.collect()


        if self.umap:
            pca = umap.UMAP(n_components=self.umap_dims, n_neighbors=len(df)-2)
            lst = [tens.numpy() for tens in df[self.inputcolumn]]
            out = pca.fit(np.asarray(lst))
            pcaOut = np.asarray([i for i in out.embedding_])
            pcaOut = 2 * (pcaOut - pcaOut.min(axis=0)) / (pcaOut.max(axis=0) - pcaOut.min(axis=0)) - 1
            # df["pca"]=[i for i in out.embedding_]
            df[self.pcacolumn] = [torch.tensor(dims) for dims in pcaOut]

        #df[self.labelcolumn] = emotions
        return df



def loadDatasetAndAddColumns(datasetPath: str, sevenEmoDs: CombinedEmoDataSet_7_emos):
    data_set = ss_encoded_dataset_full(csvPath=datasetPath)
    data_set.encoded_dataset.encodedData[data_set.encoded_dataset.dataset_column] = sevenEmoDs.dataFrame[sevenEmoDs.datasetcolumn]
    data_set.encoded_dataset.encodedData.to_pickle(datasetPath)
