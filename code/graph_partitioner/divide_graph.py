# ----------------------------------
# Import relevant modules
# ----------------------------------

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import pymetis
import networkx as nx
import torch_geometric.utils as torch_tools

# ----------------------------------
# Class with methods to split up graph into subgraphs
# ----------------------------------


class CoarseGraph:

    def __init__(self, torch_geometric_dataset, training_indices, testing_indices, number_subgraphs):
        '''
        Spilt a PyTorch Geometric dataset into a set of smaller graphs

        :param torch_geometric_dataset: dataset from the pytorch geometric library
        :param training_indices: LIST of training indices
        :param testing_indices: LIST of testing indices
        :param number_subgraphs: integer number of subgraphs to divide large graph into
        '''

        # Save input information
        if hasattr(torch_geometric_dataset, 'edge_index'):
            self.torch_geometric_dataset = torch_geometric_dataset
        else:
            print(f'[ERROR] Only compatible with PyTorch Geometric datasets for now.')

        self.training_ind = training_indices
        self.testing_ind = testing_indices
        self.number_subgraphs = number_subgraphs

        # Time it
        print(f'\n\n%------------------- METIS BEGINNING -------------------%')
        t0 = time.time()

        # ----------------------------------
        # Make the adjacency list for PyMetis
        # ----------------------------------
        # Get adjacency matrix for this graph
        adj_mat = torch.squeeze(torch_tools.to_dense_adj(self.torch_geometric_dataset.edge_index, edge_attr=self.torch_geometric_dataset.edge_attr))
        # Organize adjacency list to numpy for pymetis
        adj_list = []
        for i in range(self.torch_geometric_dataset.num_nodes):
            adj_list.append(torch.where(adj_mat[i] != 0)[0].numpy())

        # Make coarse graph
        [cost, self.parts] = pymetis.part_graph(number_subgraphs, adjacency=adj_list)
        print(f'[METIS] PyMetis cut {cost} edges to make {max(self.parts) + 1} subgraphs.')

        # ----------------------------------
        # Make subgraphs with PyMetis + testing/training indices list
        # ----------------------------------
        self.original_subgraphs = []
        self.testing_overlap = []
        self.training_overlap = []
        for i in range(max(self.parts) + 1):
            original_subgraph_i = np.argwhere(np.array(self.parts) == i).ravel().tolist()
            self.original_subgraphs.append(original_subgraph_i)
            # Get the overlap of this subgraph with the testing indices
            self.testing_overlap.append(list(set(self.testing_ind).intersection(original_subgraph_i)))
            self.training_overlap.append(list(set(self.training_ind).intersection(original_subgraph_i)))

        # ----------------------------------
        # Compute 1 Hop subgraphs for GCN: TO GET GLOBAL COMMUNICATION GRAPH
        # ----------------------------------
        self.one_hop_subgraphs = []
        self.adjacency_matrices = []
        for i in range(len(self.original_subgraphs)):
            # Compute 2 hop subgraph
            one_hop_subgraph_i = torch_tools.k_hop_subgraph(self.original_subgraphs[i], num_hops=1,
                                                    edge_index=self.torch_geometric_dataset.edge_index,
                                                    relabel_nodes=True)

            '''
            subgraph_i[0] = nodes from original graph
            subgraph_i[1] = connectivity with RELABELED indices (i.e. subgraph[0][0] -> 0, ...)
            '''
            self.one_hop_subgraphs.append(one_hop_subgraph_i[0])

        # ----------------------------------
        # Compute 2 Hop subgraphs for GCN + adjacency matrix
        # ----------------------------------
        self.testing_indices = []
        self.training_indices = []
        self.subgraphs = []
        self.adjacency_matrices = []
        for i in range(len(self.original_subgraphs)):
            # Compute 2 hop subgraph
            subgraph_i = torch_tools.k_hop_subgraph(self.original_subgraphs[i], num_hops=2,
                                                edge_index=self.torch_geometric_dataset.edge_index, relabel_nodes=True)


            '''
            subgraph_i[0] = nodes from original graph
            subgraph_i[1] = connectivity with RELABELED indices (i.e. subgraph[0][0] -> 0, ...)
            '''
            self.subgraphs.append(subgraph_i[0])

            # Adjacency matrix
            adj_mat_i = self.renormalized_laplacian(torch_tools.to_dense_adj(edge_index=subgraph_i[1]).squeeze()) # self.torch_geometric_dataset.edge_index).squeeze())
            self.adjacency_matrices.append(adj_mat_i)#[subgraph_i[0]][:, subgraph_i[0]])

            # Training and testing indices
            self.training_indices.append([subgraph_i[0].tolist().index(p) for p in self.training_overlap[i]])
            self.testing_indices.append([subgraph_i[0].tolist().index(p) for p in self.testing_overlap[i]])

        # Make the global communication matrix as well
        self.make_global_communication()
        print(f'[METIS] Subgraph creation completed in {round(time.time() - t0, 2)} seconds.')

        # Compute eigenvalues to make sure it fits our requirements
        eigs = torch.eig(self.communication_matrix)[0][:, 0].tolist()
        eigs.sort()
        self.eigs = torch.eig(self.communication_matrix, eigenvectors=True)
        print(f'[METIS] Maximum eigenvalue of communication graph: {max(eigs)}, minimum eigenvalue of communication graph: {min(eigs)}')

        # Print METIS info
        print(f'|     Subgraph    |     #edges    |     #edges+halo    |')
        print(f'|-----------------+---------------+--------------------|')
        for r in range(len(self.subgraphs)):
            print(f'|        {r}        |      {len(self.original_subgraphs[r])}      |          {len(self.subgraphs[r])}       |')

        print(f'%-------------------- METIS ENDING --------------------%\n\n')

    def renormalized_laplacian(self, adjacency_matrix):
        '''Compute renormalized weights from: https://arxiv.org/pdf/1609.02907.pdf'''

        # Make \hat{A}
        adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.shape[0])

        # Make degree matrix
        degree = torch.diag(1. / torch.sqrt(torch.sum(adjacency_matrix, dim=1)))

        adjacency_matrix = torch.matmul(degree, torch.matmul(adjacency_matrix, degree))

        return adjacency_matrix

    def make_global_communication(self):
        '''Make global connection graph based on HALO nodes'''
        # Allocate space
        self.communication_matrix1 = torch.zeros(size=(max(self.parts) + 1, max(self.parts) + 1))

        # Check for intersection
        for i in range(len(self.one_hop_subgraphs)):

            # Check all subgraphs ahead of you
            for j in range(i + 1, len(self.one_hop_subgraphs)):
                # Returns true/false indicator on overlap
                overlap_indicator = int(any(t in self.one_hop_subgraphs[j] for t in self.one_hop_subgraphs[i]))

                # Make communication graph symmetric
                self.communication_matrix1[i, j] = -overlap_indicator
                self.communication_matrix1[j, i] = -overlap_indicator

        degree = -torch.diag(torch.sum(self.communication_matrix1, dim=1))
        self.communication_matrix1 += degree
        # Get maximum eigenvalue for weighted laplacian
        max_eig = 2 * torch.max(torch.diag(degree)) + 1
        self.communication_matrix = torch.eye(max(self.parts) + 1) - self.communication_matrix1 / (2 * max_eig)

    def visualize(self, subgraph_number):
        '''Plot difference between original subgraph and 2 hop neighbor subgraph'''
        if subgraph_number > max(self.parts):
            print(f'[ERROR] Pick a subgraph number less than {max(self.parts)}')
            return

        nx_graph = torch_tools.convert.to_networkx(self.torch_geometric_dataset)

        # First subgraph
        node_labels = np.zeros(shape=(self.torch_geometric_dataset.num_nodes,))
        node_labels[self.original_subgraphs[subgraph_number]] = 1
        fig = plt.figure(1, figsize=(8, 6))
        nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color=node_labels, node_size=5, linewidths=0.01, with_labels=False)
        plt.title('First subgraph')
        plt.savefig('original.png', transparent=True)

        # Second subgraph
        node_labels = np.zeros(shape=(self.torch_geometric_dataset.num_nodes,))
        node_labels[self.subgraphs[subgraph_number]] = 1
        fig = plt.figure(1, figsize=(8, 6))
        nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color=node_labels, node_size=5, linewidths=0.01,
                with_labels=False)
        plt.title('First subgraph - 2 hop neighbors')
        plt.savefig('two_hop.png', transparent=True)




