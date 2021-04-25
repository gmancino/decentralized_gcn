# -----------------------------------------------------------------------------
#
# Graph convolutional layers from:
# https://arxiv.org/pdf/1609.02907.pdf
#
# Gabe Mancino-Ball
# -----------------------------------------------------------------------------

# ----------------------------------
# Import relevant modules
# ----------------------------------

import math
import torch
import torch.nn as nn

# ----------------------------------
# Create custom layer class
# ----------------------------------


class GraphConv(nn.Module):
    '''
    Custom Convolution layer using the Adjacency matrix of a given graph

    Inputs:
    input_dim = number of dimensions of input vector
    *** assume that each input is a ROW vector ***
    output_dim = number of dimensions returned after convolution
    '''

    def __init__(self, input_dim, output_dim):
        # Initialize the convolutional layer

        super(GraphConv, self).__init__()

        # Save dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the weights
        self.weights = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim), requires_grad=True)
        self.initialize()

    def initialize(self):
        '''Initialize parameters to be random'''
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_mat):
        '''Forward pass of this layer'''

        prod = torch.mm(x, self.weights)
        final_prod = torch.mm(adj_mat, prod)

        return final_prod