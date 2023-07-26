import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
from natsort import natsorted
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


# custom class for the chromagram transformation
class chromaTransform(object):
    def __init__(self, window='hamming', n_fft=2048, hop=512):
        self.window = window
        self.nfft = n_fft
        self.hop = hop
    
    def __call__(self, audio):
        # assumes sampling rate to be 16000 since this project uses a standardised dataset
        chroma = librosa.feature.chroma_stft(y=audio.numpy(), sr=16000, window=self.window, n_fft=self.nfft, hop_length=self.hop)
        return torch.from_numpy(chroma)


# Dataset class
class getData(Dataset):
    def __init__(self, path, frac=3, transform=None):
        self.path = path
        self.frac = frac
        self.transform = transform
        # retrieves the label from the file path of the data
        self.label = os.path.split(path)[1]
        
        # get all possible files from the data directory
        all_files = os.listdir(path)
        self.total_files = natsorted(all_files)
        print("created dataset")
    
    def __len__(self):
        # to reduce amount of data, currently taking too long to run
        return int(len(self.total_files)/self.frac)
        
    def __getitem__(self, idx):
        # get the file path of the indexed file
        file_loc = os.path.join(self.path, self.total_files[idx])
        
        # load data
        audio, sr = torchaudio.load(file_loc)
        tensor_feat = self.transform(audio)
        
        # checks the label of the data and produces an array of 1 or 0
        if self.label == "real":
            label = np.array([1], dtype='f')
        else:
            label = np.array([0], dtype='f')
        return tensor_feat[0], label

if __name__=='__main__':

    def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
            fig, axs = plt.subplots(1, 1)
            axs.set_title(title or "Spectrogram (db)")
            axs.set_ylabel(ylabel)
            axs.set_xlabel("frame")
            im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
            fig.colorbar(im, ax=axs)
            return fig
    
    def main():
        atransformer = torchaudio.transforms.MFCC(n_mfcc=128, melkwargs={"n_fft": 1024,
                                                                         "n_mels": 128,
                                                                         "hop_length": 512,
                                                                         "mel_scale": "htk",
                                                                         },)
        btransformer = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=128)
        dataset1 = getData("../Data/for-2seconds/training/real", transform=atransformer)
        dataset2 = getData("../Data/for-2seconds/training/real", transform=btransformer)
        dataloader1 = DataLoader(dataset1, batch_size=1)
        dataloader2 = DataLoader(dataset2, batch_size=1)
        audio1, label1 = next(iter(dataloader1))
        audio2, label2 = next(iter(dataloader2))
        fig = plot_spectrogram(audio2[0], title="Mel-Spectrogram")
        fig.savefig("mel_spec.png")
        fig = plot_spectrogram(audio1[0], title="MFCC")
        fig.savefig("mfcc_spec.png")
    
    main()