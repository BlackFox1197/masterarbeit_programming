import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

def spectogramWithAxis(data, sr, emotion, axis, fig):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    xdb = xdb +20
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    # x = librosa.stft(data)
    # xdb = librosa.amplitude_to_db(abs(x))
    # axis.set_title(emotion, size=20)
    # img = librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz', ax=axis, cmap=None)
    # fig.colorbar(img, ax=axis)