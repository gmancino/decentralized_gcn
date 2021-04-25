# ----------------------------------
# Import relevant modules
# ----------------------------------

from __future__ import print_function
import argparse
import os
import torch

# Custom classes
from graph_partitioner.divide_graph import CoarseGraph

# ----------------------------------
# Main method
# ----------------------------------


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Split up desired graph dataset into subgraphs.')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes.')
    parser.add_argument('--num_training_data', type=int, default=140, help='Number of training data points.')

    args = parser.parse_args()

    # ----------------------------------
    # Load Data
    # ----------------------------------
    path = os.path.join(os.getcwd(), 'data/Cora/processed/data.pt')
    dataset = torch.load(path)

    # Get training and testing indices
    train_ind = [i for i in range(args.num_training_data)]
    test_ind = [i for i in range(500, 1500)]

    data = dataset[0]

    # Gather the variables and responses
    training_data = data.x
    training_output = data.y

    # Normalize the data
    row_sum = torch.sum(training_data, dim=1)
    row_sum = 1. / row_sum
    training_data = torch.matmul(torch.diag(row_sum), training_data)

    # Use CoarseGraph to partition the dataset
    cg = CoarseGraph(data, train_ind, test_ind, args.num_nodes)

    # ----------------------------------
    # Save information to file
    # ----------------------------------
    try:
        os.remove(os.path.join(os.getcwd(), f'data/adj_matrices/global_communication.dat'))
        cg.communication_matrix.numpy().dump(os.path.join(os.getcwd(), f'data/adj_matrices/global_communication.dat'))
    except:
        cg.communication_matrix.numpy().dump(os.path.join(os.getcwd(), f'data/adj_matrices/global_communication.dat'))
    for i in range(len(cg.adjacency_matrices)):
        # Delete information
        try:
            os.remove(os.path.join(os.getcwd(), f'data/adj_matrices/adj_mat_{i:b}.dat'))
            cg.adjacency_matrices[i].numpy().dump(os.path.join(os.getcwd(), f'data/adj_matrices/adj_mat_{i:b}.dat'))
        except:
            cg.adjacency_matrices[i].numpy().dump(os.path.join(os.getcwd(), f'data/adj_matrices/adj_mat_{i:b}.dat'))

    for i in range(len(cg.subgraphs)):
        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'data/subgraph_list/subgraph_{i:b}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'data/subgraph_list/subgraph_{i:b}.txt'), 'w') as handle:
            sub_i = cg.subgraphs[i].tolist()
            for item in sub_i:
                handle.write('%s\n' % item)

    for i in range(len(cg.training_indices)):
        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'data/subgraph_list/training_inds_{i:b}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'data/subgraph_list/training_inds_{i:b}.txt'), 'w') as handle:
            for item in cg.training_indices[i]:
                handle.write('%s\n' % item)

    for i in range(len(cg.testing_indices)):
        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'data/subgraph_list/testing_inds_{i:b}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'data/subgraph_list/testing_inds_{i:b}.txt'),'w') as handle:
            for item in cg.testing_indices[i]:
                handle.write('%s\n' % item)