# -----------------------------------------------------------------------------
#
# Graph Convolutional Neural Network Implementation
# from https://github.com/tkipf/pygcn
#
# Gabe Mancino-Ball
# -----------------------------------------------------------------------------

# ----------------------------------
# Import relevant modules
# ----------------------------------

import torch
import torch.nn as nn
from gcn.layers import GraphConv

# ----------------------------------
# Create custom model class
# ----------------------------------


class GCN(nn.Module):
    '''
    Custom NN for training GCNNs using PyTorch

    Inputs:
    input_dim = number of dimensions of input vector
    hidden_dim = number of hidden dimensions
    output_dim = number of dimensions returned after completion
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):

        super(GCN, self).__init__()

        # Save inputs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Declare layers
        self.conv1 = GraphConv(self.input_dim, self.hidden_dim)
        self.conv2 = GraphConv(self.hidden_dim, self.output_dim)
        self.drop_layer = nn.Dropout(self.dropout)
        self.final_layer = nn.LogSoftmax(dim=1)

    def forward(self, x, adj_mat):
        '''Forward pass of network'''

        # Layer 1
        x = self.drop_layer(x)
        x = self.conv1(x, adj_mat)
        x = torch.relu(x)
        x = self.drop_layer(x)

        # Layer 2
        x = self.conv2(x, adj_mat)
        x = self.final_layer(x)

        return x
