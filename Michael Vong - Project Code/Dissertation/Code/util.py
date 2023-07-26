import torch
import torch.nn as nn
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import json
from pathlib import Path
from typing import Union

# MISC FUNCTIONS
'''MIT License

Copyright (c) 2022 Mark Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

matplotlib.style.use('ggplot')

# saves the best model's state
def save_checkpoint(epoch, model, optimizer, model_kwargs, model_path):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "hyper_params": model_kwargs,
    }
    torch.save(state, model_path)

# stores predictions from validation
def save_pred(y_true: np.ndarray, y_pred: np.ndarray, filename: Union[str, Path]):
    pred_to_save = {
        "y_true": np.squeeze(y_true).tolist(),
        "y_pred": np.squeeze(y_pred).tolist(),
    }
    with filename.open(mode="w") as f:
        json.dump(pred_to_save, f)
        f.close()
    return None

# plots graphs
def save_plots(
    train_acc, valid_acc, train_loss,
    acc_plot_path: Union[str, Path],
    loss_plot_path: Union[str, Path]
):
    """
    Function to save plots to disk.
    """
    # Accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    if os.path.isfile(acc_plot_path):
        os.remove(acc_plot_path)
    plt.savefig(acc_plot_path)
    
    # Loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='yellow', linestyle='-', 
        label='train loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    if os.path.isfile(loss_plot_path):
        os.remove(loss_plot_path)
    plt.savefig(loss_plot_path)


# get all possible parameter combinations in dictionary form
def paramComb(pDict):
    temp = list(pDict.keys())
    res = dict()
    cnt = 0
    
    for combinations in product(*pDict.values()):
        res[cnt] = [[ele, cnt] for ele, cnt in zip(pDict, combinations)]
        cnt += 1
    return res


# TRAINING, VALIDATION AND TESTING FUNCTIONS

# training and validation for optimisation
# paramDict here should be given a dictionary of the parameters used in random search cv
def trainVal(model, optimizer, criterion, trainLoader, valLoader, num_epochs, paramDict, model_path, best_pred_path, graph_path):
    
    # For tracking accuracy
    best_acc = 0
    tLoss = []
    tr_acc = []
    te_acc = []
    
    # Start epoch loop
    for epoch in range(num_epochs):
        # set model to train mode
        model.train()
        # tracks the loss of entire epoch
        total_loss = 0
        # tracks the number of samples
        n_total = 0.0

        # stores the predicted and true labels
        tPred = np.array([])
        tLabel = np.array([])
        
        print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
        # loop through dataloader
        for batch, data in enumerate(trainLoader):
            # extract from tuple
            feat , label = data
            # count the current size
            batchSize = feat.size(0)
            n_total += batchSize

            # get prediction
            output = model(feat)
            # calculate the loss
            loss = criterion(output, label)

            # get the predicted classes and append to the array
            predicted = np.rint(output.clone().detach().numpy())
            tPred = np.append(tPred, predicted)

            # store the matching truth label
            tLabel = np.append(tLabel, label.numpy())
            
            # calculate the running loss
            total_loss += loss.item() * batchSize

            # take backpropagation step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # get the loss of this epoch
        epoch_loss = total_loss / n_total

        # flatten the labels into 1d array
        tPred = tPred.flatten()
        tLabel = tLabel.flatten()
        # pass to sklearn accuracy function
        train_acc = accuracy_score(tLabel, tPred)
        
        # begin evaluation step
        model.eval()
        # stores the predicted and true labels for the validation step
        y_true = np.array([])
        y_pred = np.array([])
        
        # loop through dataloader
        for batch, data in enumerate(valLoader):
            feat , label = data
            # produce output from trained model
            output = model(feat)
            # store the output into the array
            predicted = np.rint(output.clone().detach().numpy())
            y_pred = np.append(y_pred, predicted)
            y_true = np.append(y_true, label.numpy())
            
        # flatten to calculate accuracy
        y_true: np.ndarray = y_true.flatten()
        y_pred: np.ndarray = y_pred.flatten()
        test_acc = accuracy_score(y_true, y_pred)
        
        # print progress
        print(f"[{epoch:03d}]: loss: {epoch_loss} - train acc: {round(train_acc, 2)} - test acc: {round(test_acc, 2)}")
        # store loss and accuracy for graph creation
        tLoss.append(epoch_loss)
        tr_acc.append(train_acc)
        te_acc.append(test_acc)


        # Store model if new one has better accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            # saves optimizer and model params as well as the parameter grid used to a .pt file
            save_checkpoint(epoch, model, optimizer, paramDict, model_path)
            # save the actual predictions made
            bestPath = Path(best_pred_path) / "best_preds.json"
            save_pred(y_true, y_pred, bestPath)
    
    # path to store graphs
    acc_path = Path(graph_path) / "accuracy.png"
    loss_path = Path(graph_path) / "loss.png"

    # save the accuracy and loss graphs
    save_plots(tr_acc, te_acc, tLoss, acc_path, loss_path)
    return best_acc
        

# Function for testing
@torch.no_grad()
def test(model, criterion, test_loader):
    # set model to eval mode
    model.eval()
    # set variable to store the loss and predictions for accuracy calculation
    n_total = 0
    total_loss = 0.0
    tPred = np.array([])
    tLabel = np.array([])

    # enumerate loader
    for batch, data in enumerate(test_loader):
        # extract from tuple
        feat , label = data
        # count the current size
        batchSize = feat.size(0)
        n_total += batchSize

        # get prediction
        output = model(feat)
        # calculate the loss
        loss = criterion(output, label)

        # get the predicted classes and append to the array
        predicted = np.rint(output.clone().detach().numpy())
        tPred = np.append(tPred, predicted)

        # store the matching truth label
        tLabel = np.append(tLabel, label.numpy())
        
        # calculate the running loss
        total_loss += loss.item() * batchSize
    
    # final loss
    total_loss = total_loss / n_total
    # final accuracy
    total_acc = accuracy_score(tLabel, tPred)
    # F1 score
    f1Score = f1_score(tLabel, tPred)

    # export classification report
    report = classification_report(tLabel, tPred)

    return total_loss, total_acc, f1Score, report


# NETWORK CLASSES

'''MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

# removes the padding on the right of the input
# input is the padding size
class Chomp(nn.Module):
    def __init__(self, size):
        super(Chomp, self).__init__()
        self.size = size
    
    def forward(self, inputs):
        return inputs[:, :, :-self.size]

# input (batch size, number of channels, sequence length)
# residual block implementation
class TempBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TempBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs,
                              kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs,
                              kernel_size, stride=stride, padding=padding, dilation=dilation)
        
        self.chomp1 = Chomp(padding)
        self.chomp2 = Chomp(padding)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        
        self.networkBlock = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                          self.conv2, self.chomp2)
        if n_inputs != n_outputs:
            self.maintainSize = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.maintainSize = None
    
    def forward(self, inputs):
        output = self.networkBlock(inputs)
        if self.maintainSize is None:
            residual = inputs
        else:
            residual = self.maintainSize(inputs)
        return self.relu2(output+residual)

# normalisation dropout and pooling module for between blocks
class regStep(nn.Module):
    def __init__(self, kernel_size, dilation, chan, dropout):
        super(regStep, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, dilation=dilation, ceil_mode=True)
        self.norm = nn.BatchNorm1d(num_features=chan)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        pooled = self.pool(inputs)
        out = self.dropout(pooled)
        outputs = self.norm(out)
        return outputs



# full network module
# reminder channel_num is a list of lists of shape [8,8,8,8]etc. where the length is the number of hidden layers
# and the number is the number of units/neurons
# input number is the number of channels in the input
class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, input_num, channel_num, k_size=3):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        levels = len(channel_num)
        for i in range(levels):
            dilation_rate = 2 ** i
            if i == 0:
                in_chan = input_num
            else:
                in_chan = channel_num[i-1]
            out_chan = channel_num[i]
            layers = layers + [TempBlock(in_chan, out_chan, k_size, stride=1, dilation=dilation_rate,
                                        padding=(k_size-1) * dilation_rate),
                                        regStep(2, dilation_rate, out_chan, 0.2)]
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.tcn(inputs)

