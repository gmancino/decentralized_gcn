# ----------------------------------
# Import relevant modules
# ----------------------------------

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

# Change fonts
import matplotlib.font_manager
from matplotlib import rc
import matplotlib.colors as mcolors
matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 16})
matplotlib.rc('text', usetex=True)

# Import custom modules
from graph_partitioner.divide_graph import CoarseGraph

# List categories
cats = ['test_accuracy', 'test_loss']
file_path = 'results'

# Read in information for the communication and computation times
# Computation
comp_1_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/computation_graphs1_epochs350_lr10.0.txt', 'r')]

comp_4_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/computation_graphs4_epochs350_lr10.0.txt', 'r')]

comp_8_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/computation_graphs8_epochs350_lr50.0.txt', 'r')]

comp_16_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/computation_graphs16_epochs350_lr100.0.txt', 'r')]

comp_32_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/computation_graphs32_epochs350_lr200.0.txt', 'r')]


# Communication
comm_1_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/communication_graphs1_epochs350_lr10.0.txt', 'r')]

comm_4_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/communication_graphs4_epochs350_lr10.0.txt', 'r')]

comm_8_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/communication_graphs8_epochs350_lr50.0.txt', 'r')]

comm_16_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/communication_graphs16_epochs350_lr100.0.txt', 'r')]

comm_32_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/communication_graphs32_epochs350_lr200.0.txt', 'r')]

# Get totals!
total_1_nodes = [comm_1_nodes[i] + comp_1_nodes[i] for i in range(len(comm_1_nodes))]
total_4_nodes = [comm_4_nodes[i] + comp_4_nodes[i] for i in range(len(comm_4_nodes))]
total_8_nodes = [comm_8_nodes[i] + comp_8_nodes[i] for i in range(len(comm_8_nodes))]
total_16_nodes = [comm_16_nodes[i] + comp_16_nodes[i] for i in range(len(comm_16_nodes))]
total_32_nodes = [comm_32_nodes[i] + comp_32_nodes[i] for i in range(len(comm_32_nodes))]

# ----------------------------------
# Plot communication and computation comparison
# ----------------------------------

# Export data to appropriate format
width = 0.3
num_bars = 5
communication_times = [sum(comm_1_nodes), sum(comm_4_nodes), sum(comm_8_nodes), sum(comm_16_nodes), sum(comm_32_nodes)]
computation_times = [sum(comp_1_nodes), sum(comp_4_nodes), sum(comp_8_nodes), sum(comp_16_nodes), sum(comp_32_nodes)]
ind = np.arange(num_bars)


fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111)
# Make border non-encompassing
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.bar(ind, communication_times, width, color=mcolors.CSS4_COLORS['skyblue'], label='communication times')
plt.bar(ind + width, computation_times, width, color=mcolors.CSS4_COLORS['steelblue'], label='computation times')
plt.xlabel('Number of GPUs')
plt.ylabel('Time (seconds)')
plt.title('Comparison of communication times and computation times')
plt.xticks(ind + width / 2, ['1 GPU', '4 GPUs', '8 GPUs', '16 GPUs', '32 GPUs'])
plt.legend()
plt.show()
plt.savefig('plots/time_comp.png', transparent=True)

# ----------------------------------
# Plot testing accuracies
# ----------------------------------

acc_1_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[0]}_graphs1_epochs350_lr10.0.txt', 'r')]

acc_4_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[0]}_graphs4_epochs350_lr10.0.txt', 'r')]

acc_8_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[0]}_graphs8_epochs350_lr50.0.txt', 'r')]

acc_16_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[0]}_graphs16_epochs350_lr100.0.txt', 'r')]

acc_32_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[0]}_graphs32_epochs350_lr200.0.txt', 'r')]

fig1 = plt.figure(figsize=(8, 6.5))
ax = fig1.add_subplot(111)
# Make border non-encompassing
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.hlines(81.5, 0, 350, color=mcolors.CSS4_COLORS['lightcoral'], linestyles='dashed', label='benchmark')
plt.plot(acc_1_nodes, color=mcolors.CSS4_COLORS['maroon'], label='1 GPU')
plt.plot(acc_4_nodes, color=mcolors.CSS4_COLORS['indigo'], label='4 GPUs')
plt.plot(acc_8_nodes, color=mcolors.CSS4_COLORS['darkorange'], label='8 GPUs')
plt.plot(acc_16_nodes, color=mcolors.CSS4_COLORS['royalblue'], label='16 GPUs')
plt.plot(acc_32_nodes, color=mcolors.CSS4_COLORS['forestgreen'], label='32 GPUs')
plt.xlabel('Number of data passes')
plt.ylabel('Accuracy (percent)')
plt.title('Testing accuracy')
plt.legend()
plt.show()
plt.savefig('plots/acc.png', transparent=True)

# ----------------------------------
# Plot testing loss
# ----------------------------------

loss_1_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[1]}_graphs1_epochs350_lr10.0.txt', 'r')]

loss_4_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[1]}_graphs4_epochs350_lr10.0.txt', 'r')]

loss_8_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[1]}_graphs8_epochs350_lr50.0.txt', 'r')]

loss_16_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[1]}_graphs16_epochs350_lr100.0.txt', 'r')]

loss_32_nodes = [float(line.rstrip('\n')) for line in
          open(f'{file_path}/{cats[1]}_graphs32_epochs350_lr200.0.txt', 'r')]

fig2 = plt.figure(figsize=(8, 6.5))
ax = fig2.add_subplot(111)
# Make border non-encompassing
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(loss_1_nodes, color=mcolors.CSS4_COLORS['maroon'], label='1 GPU')
plt.plot(loss_4_nodes, color=mcolors.CSS4_COLORS['indigo'], label='4 GPUs')
plt.plot(loss_8_nodes, color=mcolors.CSS4_COLORS['darkorange'], label='8 GPUs')
plt.plot(loss_16_nodes, color=mcolors.CSS4_COLORS['royalblue'], label='16 GPUs')
plt.plot(loss_32_nodes, color=mcolors.CSS4_COLORS['forestgreen'], label='32 GPUs')
plt.xlabel('Number of data passes')
plt.ylabel('Loss')
plt.title('Testing loss')
plt.legend()
plt.show()
plt.savefig('plots/loss.png', transparent=True)

# ----------------------------------
# Time versus accuracy
# ----------------------------------


def find_first_consecutive(array, look_back_distance):
    '''Function to find the first point in a list that the next look_back_distance numbers follow consecutively'''
    # Keep track of consecutive numbers
    j = 0

    for i in range(len(array) - 1):

        if array[i+1] == (array[i] + 1):
            j += 1
        else:
            j = 0

        if j == look_back_distance:
            return array[i - look_back_distance]
        else:
            pass

    return None


time_1_node = sum(total_1_nodes[0: find_first_consecutive(np.where(np.array(acc_1_nodes) >= 81.5)[0].tolist(), 10)])
time_4_node = sum(total_4_nodes[0: find_first_consecutive(np.where(np.array(acc_4_nodes) >= 81.5)[0].tolist(), 10)])
time_8_node = sum(total_8_nodes[0: find_first_consecutive(np.where(np.array(acc_8_nodes) >= 81.5)[0].tolist(), 10)])
time_16_node = sum(total_16_nodes[0: find_first_consecutive(np.where(np.array(acc_16_nodes) >= 81.5)[0].tolist(), 10)])
time_32_node = sum(total_32_nodes[0: find_first_consecutive(np.where(np.array(acc_32_nodes) >= 81.5)[0].tolist(), 10)])

fig3 = plt.figure(figsize=(8, 6.5))
ax = fig3.add_subplot(111)
# Make border non-encompassing
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.hlines(4, 1, 6, color=mcolors.CSS4_COLORS['lightcoral'], linestyles='dashed', label='benchmark')
plt.bar(1, 4 / time_1_node, color=mcolors.CSS4_COLORS['maroon'])
plt.bar(2, 4 / time_4_node, color=mcolors.CSS4_COLORS['indigo'])
plt.bar(3, 4 / time_8_node, color=mcolors.CSS4_COLORS['darkorange'])
plt.bar(4, 4 / time_16_node, color=mcolors.CSS4_COLORS['royalblue'])
plt.bar(5, 4 / time_32_node, color=mcolors.CSS4_COLORS['forestgreen'])
plt.xlabel('Number of GPUs')
plt.ylabel('Speed up')
plt.title('Speed up factor (relative to time to reach benchmark arccuracy)')
plt.xticks(np.arange(1, 6, 1), ['1 GPU', '4 GPUs', '8 GPUs', '16 GPUs', '32 GPUs'])
plt.show()
plt.savefig('plots/time_speedup.png', transparent=True)

fig4 = plt.figure(figsize=(8, 6.5))
ax = fig4.add_subplot(111)
# Make border non-encompassing
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.hlines(4, 0, 6, color=mcolors.CSS4_COLORS['lightcoral'], linestyles='dashed', label='benchmark')
plt.bar(1, time_1_node, color=mcolors.CSS4_COLORS['maroon'])
plt.bar(2, time_4_node, color=mcolors.CSS4_COLORS['indigo'])
plt.bar(3, time_8_node, color=mcolors.CSS4_COLORS['darkorange'])
plt.bar(4, time_16_node, color=mcolors.CSS4_COLORS['royalblue'])
plt.bar(5, time_32_node, color=mcolors.CSS4_COLORS['forestgreen'])
plt.xlabel('Number of GPUs')
plt.ylabel('Time (seconds)')
plt.title('Time taken to reach benchmark accuracy')
plt.xticks(np.arange(1, 6, 1), ['1 GPU', '4 GPUs', '8 GPUs', '16 GPUs', '32 GPUs'])
plt.show()
plt.savefig('plots/time_to_acc.png', transparent=True)

# ----------------------------------
# Visualize sparsity patterns of communication graph
# ----------------------------------

num_gpus = [4, 8, 16, 32]

dataset = Planetoid(root='data', name='Cora')

# Get training and testing indices
train_ind = [i for i in range(140)]
test_ind = [i for i in range(500, 1500)]

data = dataset[0]

# Gather the variables and responses
training_data = data.x
training_output = data.y

# FOR LOOP OVER EACH PARTITION
comm_matrices = []
for i in range(len(num_gpus)):

    # Use CoarseGraph to partition the dataset
    cg = CoarseGraph(data, train_ind, test_ind, num_gpus[i])

    # Communication matrix
    comm_matrices.append(cg.communication_matrix.numpy())


# Create sparisty plot
fig5, axs = plt.subplots(2, 2, figsize=(8, 6.5))
# Make border non-encompassing
# axs.spines['right'].set_visible(False)
# axs.spines['top'].set_visible(False)
axs[0, 0].spy(comm_matrices[0], markersize=10, color=mcolors.CSS4_COLORS['black'])
axs[0, 0].set_title('4 GPUs')
axs[0, 0].axis('off')
axs[0, 1].spy(comm_matrices[1], markersize=5, color=mcolors.CSS4_COLORS['black'])
axs[0, 1].set_title('8 GPUs')
axs[0, 1].axis('off')
axs[1, 0].spy(comm_matrices[2], markersize=2, color=mcolors.CSS4_COLORS['black'])
axs[1, 0].set_title('16 GPUs')
axs[1, 0].axis('off')
axs[1, 1].spy(comm_matrices[3], markersize=1, color=mcolors.CSS4_COLORS['black'])
axs[1, 1].set_title('32 GPUs')
axs[1, 1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig('plots/sparsity.png', transparent=True)
