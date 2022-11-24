import librosa.display
from matplotlib import pyplot as plt


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    # librosa.display.waveplot(data, sr=sr)
    # plt.show


def waveplot_two_datas(data, sr, data1, sr1, emotion, emotion1):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))
    ax[0].set_title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr, ax=ax[0])
    ax[1].set_title(emotion1, size=20)
    librosa.display.waveshow(data1, sr=sr1, ax=ax[1])

