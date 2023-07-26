import dataLoad
import CreateModel
import melSpecModel
import util

import os

import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

'''
This file is where the code should be run from
To test on the best parameter combination found, run this file as it is
To perform parameter search again, delete the .pt files in the "trained_models" folder
'''

torch.backends.openmp.is_available()
# seed random
torch.manual_seed(2022)
torch.get_num_threads
torch.set_num_threads(1)

# path for trained model state
model_path = "../trained_models/melspectrogram/best_model.pth"
new_best_path = "../trained_models/melspectrogram/re_best_model.pth"

# path to save testing results
res_path = "../Results/melspectrogram/testResult.txt"

# best predictions
best_pred_path = "../Results/melspectrogram/"

# graph directory
image_path = "../image/melspectrogram/"

# data file paths
tRealPath = "../Data/for-2seconds/training/real"
tFakePath = "../Data/for-2seconds/training/fake"

vRealPath = "../Data/for-2seconds/validation/real"
vFakePath = "../Data/for-2seconds/validation/fake"

realPath = "../Data/for-2seconds/testing/real"
fakePath = "../Data/for-2seconds/testing/fake"

# read parameters from training if already done, else run training
if os.path.isfile(model_path):
    trained = torch.load(model_path)
    params = trained["hyper_params"]
else:
    print("start tuning")
    melSpecModel.mspecTuning()
    print("finish tuning")
    trained = torch.load(model_path)
    params = trained["hyper_params"]

# setup transform function
mspec_transformer = torchaudio.transforms.MelSpectrogram(n_fft=params['nfft'], n_mels=params['n_mels'])

# setup dataset and data loader
# Training set
realDataset = dataLoad.getData(tRealPath, frac=1, transform=mspec_transformer)
fakeDataset = dataLoad.getData(tFakePath, frac=1, transform=mspec_transformer)
trainDataset = torch.utils.data.ConcatDataset([realDataset, fakeDataset])
# Validation set
vrealDataset = dataLoad.getData(vRealPath, frac=1, transform=mspec_transformer)
vfakeDataset = dataLoad.getData(vFakePath, frac=1, transform=mspec_transformer)
valDataset = torch.utils.data.ConcatDataset([vrealDataset, vfakeDataset])
# Test set
realSet = dataLoad.getData(realPath, frac=1, transform=mspec_transformer)
fakeSet = dataLoad.getData(fakePath, frac=1, transform=mspec_transformer)
dataSet = trainDataset = torch.utils.data.ConcatDataset([realSet, fakeSet])


# set up dataloaders
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=32, shuffle=True)
loader = DataLoader(dataSet, batch_size=32, shuffle=True)

# determine input size
inSize = [len(a) for a in dataSet[0]][0]
# setup parameters to store results
kSize = 3
num_epochs = 10

# setup model and criterion
model = CreateModel.tcnClassifier(inSize, 1, num_channels=params['num_channels'], kernel_size=kSize)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

# begin training
val_acc = util.trainVal(model, optimizer, criterion, trainLoader, valLoader, num_epochs, params, new_best_path, best_pred_path, image_path)
print(val_acc)

# get retrained model state
retrained = torch.load(new_best_path)
model.load_state_dict(retrained["state_dict"])

print("starting test on retrained model")
# evaluate model
loss, acc, f1, report = util.test(model, criterion, loader)
print("ReTrained Model Loss: ", loss)
print("ReTrained Model Accuracy: ", acc)
print("ReTrained Model F1 Score: ", f1)
print(report)

# save results
with open(res_path, 'w') as f:
    f.write('Loss = %s\n Accuracy = %s\n F1 Score = %s\n' % (loss, acc, f1))
    f.write(report)
    f.close()