import dataLoad
import CreateModel
import util

import random
import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(2022)
random.seed(2022)
torch.set_num_threads(1)
'''
This is the model using melspectrograms as features
'''
def mspecTuning():
    print("MELSPECTROGRAM FEATURES")
    
    # file paths
    tRealPath = "../Data/for-2seconds/training/real"
    tFakePath = "../Data/for-2seconds/training/fake"
    
    vRealPath = "../Data/for-2seconds/validation/real"
    vFakePath = "../Data/for-2seconds/validation/fake"
    
    # model state dictionary path
    model_path = "../trained_models/melspectrogram/best_model.pth"
    
    # best predictions
    best_pred_path = "../Results/melspectrogram/"

    # graph directory
    image_path = "../image/melspectrogram/"
    
    # parameter grid
    parameters = {"nfft" : [1024, 2048],
                  "n_mels" : [128, 64],
                  "lr" : [1e-3, 1e-4, 1e-5],
                  "weight_decay" : [1e-5, 1e-6, 1e-7]}
                  
    paramGrid = util.paramComb(parameters)
    
    # generate random 10 indexes for parameter combinations
    num_comb = random.sample(range(0,len(paramGrid)), 10)
    
    # To store id for best parameter combinations
    key_list = ["nfft", "n_mels", "num_channels", "lr", "weight_decay"]
    best_combi = dict(zip(key_list, [None]*len(key_list)))
    best_vacc = 0.0
    
    # grid parameters loop
    for paramID in num_comb:
        # set up params
        params = paramGrid[paramID]
        nfft = params[0][1]
        n_mels = params[1][1]
        lr = params[2][1]
        weight_decay = params[3][1]
        num_channels = [n_mels]*2
        
        # for saving if best
        paramDict = {"nfft" : nfft,
                     "n_mels" : n_mels,
                     "num_channels" : num_channels,
                     "lr" : lr,
                     "weight_decay" : weight_decay}
        
        # set kernel size
        kSize = 3
        # set number of epochs
        num_epochs = 10
        
        # setup transformer to convert to mel spectrograms
        mspec_transformer = torchaudio.transforms.MelSpectrogram(n_fft=nfft, n_mels=n_mels)
        
        # loads all the data to prepare
        # Training set
        realDataset = dataLoad.getData(tRealPath, frac=1, transform=mspec_transformer)
        fakeDataset = dataLoad.getData(tFakePath, frac=1, transform=mspec_transformer)
        trainDataset = torch.utils.data.ConcatDataset([realDataset, fakeDataset])
        # Validation set
        vrealDataset = dataLoad.getData(vRealPath, frac=2, transform=mspec_transformer)
        vfakeDataset = dataLoad.getData(vFakePath, frac=2, transform=mspec_transformer)
        valDataset = torch.utils.data.ConcatDataset([vrealDataset, vfakeDataset])
        
        # set up dataloaders
        trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=32, shuffle=True)
        print("Data loaders created")
        # determine input size
        inSize = [len(a) for a in trainDataset[0]][0]
        # make network
        model = CreateModel.tcnClassifier(inSize, 1, num_channels=num_channels, kernel_size=kSize)
        
        # optimiser and criterion
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # perform training and validation
        val_acc = util.trainVal(model, optimizer, criterion, trainLoader, valLoader, num_epochs, paramDict, model_path, best_pred_path, image_path)
        
        # check for best version and save the parameters
        if val_acc > best_vacc:
            best_combi.update(paramDict)
            best_vacc = val_acc
    
    return None

if __name__=='__main__':
    mspecTuning()