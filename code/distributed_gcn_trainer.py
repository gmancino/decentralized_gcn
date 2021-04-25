# ----------------------------------
# Import relevant modules
# ----------------------------------

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import math
from mpi4py import MPI
from torch_geometric.datasets import Reddit

# Custom classes
from gcn.models import GCN
from gcn.replace_weights import Opt

# ----------------------------------
# MPI set-up
# ----------------------------------

comm = MPI.COMM_WORLD
size = comm.Get_size()
# Get the current processor!
rank = comm.Get_rank()
# Attach buffer
BUFF_SISE = 32064000 * (1 + MPI.BSEND_OVERHEAD)
buff = numpy.empty(BUFF_SISE, dtype='b')
MPI.Attach_buffer(buff)


# ---------------------------------- #
#                 DGD                #
# ---------------------------------- #

class DGD:
    '''
    Class for solving decentralized nonconvex consensus problems.

    :param
    '''

    def __init__(self, local_params, mixing_matrix, data, labels):

        # -------- GATHER COMMUNICATION INFORMATION FROM THE MIXING MATRIX -------- #
        self.mixing_matrix = mixing_matrix.float()
        self.num_nodes = self.mixing_matrix.shape[0]

        # -------- PARSE COMMUNICATION GRAPH TO GET PEERS AND CORRESPONDING WEIGHTS -------- #
        self.peers = torch.where(self.mixing_matrix[rank, :] != 0)[0].tolist()
        # Remove yourself from the list
        self.peers.remove(rank)

        # Print number of neighbors
        print(f'[Rank {rank} number of neighbors] {len(self.peers)}')

        # Get the weights
        self.peer_weights = self.mixing_matrix[rank, self.peers].tolist()

        # Get weights
        self.my_weight = self.mixing_matrix[rank, rank].item()

        # -------- PARSE INPUT/TRAINING PARAMETERS -------- #
        if 'local_mixing_matrix' in local_params:
            self.local_mixing_matrix = local_params['local_mixing_matrix']
        else:
            self.local_mixing_matrix = torch.ones(self.num_nodes)
        if 'local_nodes' in local_params:
            self.local_nodes = local_params['local_nodes']
        else:
            self.local_nodes = torch.ones(self.num_nodes)
        if 'train_inds' in local_params:
            self.train_inds = local_params['train_inds']
        else:
            self.train_inds = [i for i in range(100)]
        if 'lr' in local_params:
            self.lr = local_params['lr']
        else:
            self.lr = 1e-4
        if 'dropout' in local_params:
            self.dropout = local_params['dropout']
        else:
            self.dropout = 0.5
        if 'hidden' in local_params:
            self.hidden = local_params['hidden']
        else:
            self.hidden = 64

        # -------- GET THE CUDA DEVICE -------- #
        self.device = torch.device(f'cuda:{rank % 8}')

        # -------- SPLIT THE DATA -------- #
        self.data = data[self.local_nodes, :].to(self.device)
        self.labels = labels[self.local_nodes].to(self.device)
        self.local_mixing_matrix = self.local_mixing_matrix.to(self.device)

        # Get problem dimension
        self.problem_dim = self.data.shape[1]

        # -------- INITIALIZE THE APPROPRIATE FUNCTIONS -------- #
        # Objective
        self.num_classes = max(labels).item() + 1
        self.objective_function = GCN(self.data.shape[1], self.hidden, self.num_classes, dropout=self.dropout).to(self.device)

        # Initialize the updating weights rule
        self.replace_weights = Opt(self.objective_function.parameters(), lr=0.1)
        # Initialize training loss
        self.training_loss = torch.nn.NLLLoss(reduction='mean')

        # -------- INITIALIZE THE TESTING FUNCTIONS -------- #
        self.testing_function = GCN(self.data.shape[1], self.hidden, self.num_classes, dropout=self.dropout).to(self.device)
        self.testing_optimizer = Opt(self.testing_function.parameters(), lr=1e-2)

        # -------- CREATE VARIABLES ON SEPARATE GPUS -------- #
        self.weights = [torch.randn(size=p.shape).to(self.device) * math.sqrt((1 / p.shape[0])) for p in self.objective_function.parameters()]

        # Initialize dual variables (both sets!)
        self.grads = [torch.zeros(size=p.shape).to(self.device) for p in self.objective_function.parameters()]

        # Save number of parameters
        self.num_params = len(self.weights)

        # Save norm histories and consensus histories
        self.consensus_violation = []
        self.norm_hist = []

        # -------- ALLOCATE SPACE FOR TESTING RESULTS -------- #
        self.testing_loss = []
        self.testing_accuracy = []

        # -------- ALLOCATE SPACE FOR TIMING RESULTS -------- #
        self.compute_time = []
        self.communication_time = []

    def solve(self, outer_iterations, test_inds):
        '''Solve the global problem'''

        # TIME IT
        t0 = time.time()

        # Barrier communication at beginning of run
        comm.Barrier()

        # Loop over algorithm updates
        for i in range(outer_iterations):

            # TIME THIS EPOCH
            time_i = time.time()

            # GET THE GRADIENTS AT THE CURRENT VALUE
            self.grads = self.get_grads(self.weights)

            # STOP TIME FOR COMPUTING
            int_time1 = time.time()

            # ----- PERFORM COMMUNICATION ----- #
            comm_time = self.communicate_with_neighbors()
            comm.Barrier()
            # ---------------------------------- #

            # STOP TIME FOR COMPUTING
            int_time2 = time.time()

            # UPDATE THE WEIGHTS
            self.weights = [self.weights[j] - (self.lr) * self.grads[j] for j in range(self.num_params)]

            # END TIME
            time_i_end = time.time()
            comp_time = round(time_i_end - int_time2 + int_time1 - time_i, 4)

            # Test the model
            if rank == 0:
                print(f'\n% --------------- Iteration {i}: total time = {comp_time} s, comm time = {comm_time} s --------------- %')

            comm.Barrier()
            test_loss, test_acc = self.test(self.weights, test_inds)
            self.testing_loss.append(test_loss)
            self.testing_accuracy.append(test_acc)

            # APPEND TIMING INFORMATION
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)

        # END TIME
        t1 = time.time() - t0
        if rank == 0:
            print(f'% --------------- TOTAL TRAINING TIME: {round(t1, 2)} s --------------- %\n\n')

        # Return the training time
        return t1

    def communicate_with_neighbors(self):

        # TIME IT
        time0 = time.time()

        # ----- LOOP OVER PARAMETERS ----- #
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.weights[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=(len(self.peers) * self.weights[pa].shape[0], self.weights[pa].shape[1]), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for ind in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[(ind * self.weights[pa].shape[0]):((ind + 1) * self.weights[pa].shape[0]), :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            self.weights[pa] = self.my_weight * self.weights[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.weights[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[(ind * self.weights[pa].shape[0]):((ind + 1) * self.weights[pa].shape[0]), :]).to(self.device))

        return round(time.time() - time0, 4)

    def test(self, weights, test_inds):
        '''Test the data using the global weights'''

        self.testing_optimizer.step(weights, self.device)

        # Change to evaluation mode
        self.testing_function.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0
        num_test_nodes = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():

            # Evaluate the model on the testing data
            output = self.testing_function(self.data, self.local_mixing_matrix)

            # Update numnber of nodes
            num_test_nodes += len(test_inds)

            # Sum batch loss
            test_loss += loss_function(output[test_inds], self.labels[test_inds]).item()

            # Gather predictions on testing data
            pred = torch.argmax(output[test_inds], dim=1)
            correct += torch.sum(pred == self.labels[test_inds]).item()

        # COMMUNICATE TO 1 NODE
        if rank != 0:
            array_to_send = numpy.array([correct, num_test_nodes, test_loss])
            comm.Send(array_to_send, dest=0)
        else:
            for t in range(1, self.num_nodes):
                rcv = numpy.empty((3,))
                status = MPI.Status()
                comm.Recv(rcv, source=MPI.ANY_SOURCE, status=status)
                correct += rcv[0]
                num_test_nodes += rcv[1]
                test_loss += rcv[2]

        comm.Barrier()

        # Save loss and accuracy
        test_loss /= num_test_nodes
        testing_accuracy = 100. * correct / num_test_nodes

        if rank == 0:
            print(f'[Info] Test Set: Average Loss {round(test_loss, 6)}, Accuracy: '
                f'{int(correct)}/{int(num_test_nodes)} ({round(testing_accuracy, 2)} %)\n')

        return test_loss, testing_accuracy

    def get_grads(self, init_guess):
        '''Solve the local problem by ADAM'''

        # Take the first step
        self.replace_weights.zero_grad()
        self.replace_weights.step(init_guess, self.device)

        # Forward pass of the model
        out = self.objective_function(self.data, self.local_mixing_matrix)
        loss = (1 / self.num_nodes) * self.training_loss(out[self.train_inds, :], self.labels[self.train_inds])

        # Compute the gradients
        loss.backward()

        return [p.grad.data.detach().to(self.device) for p in self.objective_function.parameters()]


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='DGD testing on GCN problem for differing communication graphs.')

    parser.add_argument('--epochs', type=int, default=350, help='Total number of communication rounds.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Local learning rate.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout for regularization.')
    parser.add_argument('--hidden', type=int, default=64, help='Dimension of hidden layer.')
    parser.add_argument('--data', type=str, default='Cora', choices=('Cora', 'Reddit'), help='Graph dataset to use.')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes.')
    parser.add_argument('--num_training_data', type=int, default=140, help='Number of training data points.')

    args = parser.parse_args()

    # ----------------------------------
    # Load Data
    # ----------------------------------
    print(f'Rank {rank} checking in!')

    if args.data == 'Cora':
        if rank == 0:
            print(f'[INFO] Loading CORA dataset...\n')
        path = os.path.join(os.getcwd(), 'data/Cora/processed/data.pt')
        print(f'Path to data {path}')
        dataset = torch.load(path) # Planetoid(root='data', name='Cora')

    elif args.data == 'Reddit':
        if rank == 0:
            print(f'[INFO] Loading REDDIT dataset...\n')
        dataset = Reddit(root='data')
    else:
        print(f'[ERROR] Please choose a valid dataset: Cora or Reddit.')
        exit(-1)

    data = dataset[0]

    # Gather the variables and responses
    training_data = data.x
    training_output = data.y

    # Normalize the data
    row_sum = torch.sum(training_data, dim=1)
    row_sum = 1. / row_sum
    training_data = torch.matmul(torch.diag(row_sum), training_data)

    # GATHER DATA FROM TXT FILES AND FROM NUMPY MATRICES
    communication_matrix = torch.tensor(numpy.load(os.path.join(os.getcwd(), f'data/adj_matrices/global_communication.dat'), allow_pickle=True))
    adj_matrix = torch.tensor(numpy.load(os.path.join(os.getcwd(), f'data/adj_matrices/adj_mat_{rank:b}.dat'), allow_pickle=True))
    subgraph_nodes = [float(line.rstrip('\n')) for line in open(os.path.join(os.getcwd(), f'data/subgraph_list/subgraph_{rank:b}.txt'), 'r')]
    training_inds = [float(line.rstrip('\n')) for line in open(os.path.join(os.getcwd(), f'data/subgraph_list/training_inds_{rank:b}.txt'), 'r')]
    testing_inds = [float(line.rstrip('\n')) for line in open(os.path.join(os.getcwd(), f'data/subgraph_list/testing_inds_{rank:b}.txt'), 'r')]

    # ----------------------------------
    # Print training info
    # ----------------------------------
    if rank == 0:
        print(f'\n\n%--------------- BEGINNING TRAINING ---------------%')
        print(f'[INFO] Hidden dimension size: {args.hidden}\n\n')

    # COMMUNICATION BARRIER
    comm.Barrier()

    # ----------------------------------
    # Train the model
    # ----------------------------------
    algo_params = {'lr': args.lr, 'local_mixing_matrix': adj_matrix,
                   'local_nodes': subgraph_nodes, 'train_inds': training_inds, 'dropout': args.dropout,
                   'hidden': args.hidden}

    solver = DGD(algo_params, communication_matrix, training_data, training_output)

    algo_time = solver.solve(args.epochs, testing_inds)

    # ----------------------------------
    # Save the information
    # ----------------------------------
    if rank == 0:
        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'results/info_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        with open(os.path.join(os.getcwd(), f'results/info_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'), 'w') as handle:
            handle.write(f'% -------------------------- TRAINING INFORMATION -------------------------- %\n')
            handle.write(f'[DATASET] {args.data}\n')
            handle.write(f'[NUM NODES] {args.num_nodes}\n')
            handle.write(f'[EPOCHS] {args.epochs}\n')
            handle.write(f'[CONNECTION GRAPH] maximum eig: {torch.max(torch.eig(communication_matrix)[0][:, 0]).item()}, minimum eig: {torch.min(torch.eig(communication_matrix)[0][:, 0]).item()}\n')
            handle.write(f'[LOCAL LEARNING RATE] {args.lr}\n')
            handle.write(f'[NUM TRAINING DATA] {args.num_training_data}\n')
            handle.write(f'[TOTAL TRAINING TIME] {algo_time}\n')
            handle.write(f'% -------------------------------------------------------------------------- %')

        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'results/test_loss_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'results/test_loss_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                  'w') as handle:
            for item in solver.testing_loss:
                handle.write('%s\n' % item)

        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'results/test_accuracy_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'), 'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'results/test_accuracy_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                  'w') as handle:
            for item in solver.testing_accuracy:
                handle.write('%s\n' % item)

        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'results/communication_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                        'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'results/communication_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                  'w') as handle:
            for item in solver.communication_time:
                handle.write('%s\n' % item)
        # Delete information
        try:
            file = open(os.path.join(os.getcwd(), f'results/computation_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                        'r+')
            file.truncate(0)
            file.close()
        except:
            pass
        # Rewrite information
        with open(os.path.join(os.getcwd(), f'results/computation_graphs{args.num_nodes}_epochs{args.epochs}_lr{args.lr}.txt'),
                  'w') as handle:
            for item in solver.compute_time:
                handle.write('%s\n' % item)

    # BARRIER
    comm.Barrier()