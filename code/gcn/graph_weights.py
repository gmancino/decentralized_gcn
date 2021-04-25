# -----------------------------------------------------------------------------
#
# Create different Adjacency weightings for graph data
#
# Gabe Mancino-Ball
# -----------------------------------------------------------------------------

# ----------------------------------
# Import relevant modules
# ----------------------------------

import torch
import numpy

# ----------------------------------
# Create custom adjacency matrix
# ----------------------------------


class AdjacencyMatrix:
    '''
    Custom adjacency matrix given the list of connections

    Inputs:
    connectivity = [2, num_edges] tensor where each column indicates an (undirected) edge
    num_nodes = number of nodes in the graph

    Outputs:
    adjacency_matrix = adjacency matrix
    '''

    def __init__(self, connectivity, num_nodes):

        # Get connectivity
        if connectivity.shape[0] == 2:
            self.connectivity = self.conv_to_tensor(connectivity)
        else:
            print(f'[Info] Cannot make adjacency matrix with {connectivity.shape[0]} sets of edges.')
            return

        # Get number of nodes
        self.num_nodes = num_nodes

        # Set diagonals to 1 to ensure each node shares with itself
        self.adjacency_matrix = torch.diag(torch.ones(self.num_nodes))

        # Set edges as actual adjacency matrix
        self.adjacency_matrix[self.connectivity[0], self.connectivity[1]] = 1

    def metropolis_weights(self):
        '''Create a metropolis weight matrix based on the graph connectivty'''

        # Subtract 1 to only show outbound connectivities
        graph_degree = torch.sum(self.adjacency_matrix, dim=0) - 1

        # Check node-wise max degree
        graph_degree_transpose = torch.transpose(graph_degree[None, :], 0, 1)
        max_degree = torch.max(graph_degree, graph_degree_transpose)

        # Create the outbound and inbound weights
        weights = 1 / max_degree

        # Create appropriate connections
        weights = weights * self.adjacency_matrix

        # Create diagonals
        row_sums = 1 - torch.sum(weights, dim=0)
        self.adjacency_matrix = weights + torch.diag(row_sums)

    def laplacian_weights(self):
        '''Use the graph Laplacian to make the weights for the graph'''

        # Subtract 1 to only show outbound connectivities
        graph_degree = torch.sum(self.adjacency_matrix, dim=0) - 1

        # Create graph laplacian
        self.adjacency_matrix = torch.diag(graph_degree) - self.adjacency_matrix - torch.eye(self.num_nodes)

    def renormalized_weights(self):
        '''Graph weights are made from this paper: https://arxiv.org/pdf/1609.02907.pdf'''

        # Get degree matrix of A+I
        graph_degree = torch.sum(self.adjacency_matrix, dim=0)

        # Create diagonal matrix
        normalized_degree_mat = torch.diag(1 / torch.sqrt(graph_degree))

        # Create renormalized adjacency matrix
        self.adjacency_matrix = torch.matmul(torch.matmul(normalized_degree_mat, self.adjacency_matrix), normalized_degree_mat)

    def conv_to_tensor(self, input):
        '''Convert input to Torch tensor'''

        if isinstance(input, numpy.ndarray):
            input_tensor = torch.tensor(input, dtype=torch.float)

        elif isinstance(input, torch.Tensor):
            input_tensor = input

        else:
            print('[Info] Please input a Numpy array or a Torch tensor.')
            input_tensor = False

        return input_tensor
