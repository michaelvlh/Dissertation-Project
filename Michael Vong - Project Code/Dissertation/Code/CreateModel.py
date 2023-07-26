import torch
import torch.nn as nn
from torch.nn import functional as F
from util import TemporalConvolutionNetwork

# combines the convolutional module with a fully connected layer to produce class probabilities
# num_channels is a list integers of length x. Where x is the number of blocks and the integers represent the number of neurons for each layer or block.
# out_size should be set to 1 due to the task being binary
class tcnClassifier(nn.Module):
    def __init__(self, in_size, out_size, num_channels, kernel_size):
        super(tcnClassifier, self).__init__()
        self.tcn = TemporalConvolutionNetwork(in_size, num_channels, k_size=kernel_size)
        self.dense1 = nn.LazyLinear(num_channels[-1])
        self.dense2 = nn.Linear(num_channels[-1], out_size)
        self.flat = nn.Flatten()
        
    def forward(self, inputs):
        convOut = self.tcn(inputs)
        flat = self.flat(convOut)
        out = F.relu(self.dense1(flat))
        out = self.dense2(out)
        return torch.sigmoid(out)