# Decentralized training of graph convolutional networks
---

This is the code and report for my final project in Dr. Carother's _Parallel Computing_ course taught in the Spring of 2021 at RPI.

## Project overview

Graph Convolutional Networks ([GCNs](https://arxiv.org/pdf/1609.02907.pdf)) can be used to solve node level classification problems in graph structured data. In a node level classification problem, we assume a given dataset has an underlying graph structure where each node in the graph contains a data point and _some_ nodes (i.e. a small percentage) contain corresponding labels. The goal is to learn a classifier that performs well on nodes with unknown labels. As the size of graph datasets grow, the need to perform distributed training of GCNs is necessitated. This work applies concepts from decentralized consensus optimization, namely the Decentralized Gradient Descent ([DGD](https://arxiv.org/pdf/1608.05766.pdf)) method, to assist in training GCNs for graph structured data. We perform numerical experiments on the [AiMOS](https://cci.rpi.edu/aimos) supercomputer at RPI and include a [technical report](mancino_ball_parallel_project.pdf).

## What's in the repo

+ `code`: folder containing the code used to produce the [technical report](mancino_ball_parallel_project.pdf) associated with this repository
  + `data`: folder containing the [Cora](https://graphsandnetworks.com/the-cora-dataset/) dataset and empty folders to store the partitioned data
  + `gcn`: folder containing a custom implementation of the [GCN](https://arxiv.org/pdf/1609.02907.pdf) architecture used for solving node classification problems
  + `graph_partitioner`: folder containing the code that partitions a graph dataset among $N$ agents using [PyMetis](https://pypi.org/project/PyMetis/)
  + `distributed_gcn_trainer.py`: python file that performs [DGD](https://arxiv.org/pdf/1608.05766.pdf) using [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
  + `make_plots.py`: python file that produces the figures in the [technical report](mancino_ball_parallel_project.pdf)
+ `requirements.txt`: a text file containing the packages required in the [conda](https://docs.conda.io/en/latest/miniconda.html) environment that a user sets up to run these experiments

## How to run on AiMOS

[AiMOS](https://cci.rpi.edu/aimos) is the supercomputer located on RPI's campus. As usernames and project names are sensitive material, below is a mock `.sh` file script that would need to be located in the `code` folder to run this script with multiple GPUs.

**Note:** anthing in `<>` brackets is considered a user argument and _must_ be specified before running on AiMOS.

```
#!/bin/bash -x

# ----- GATHER SLURM INFORMATION FOR RUNNING ON MULTIPLE COMPUTE NODES ----- #
if [ "x$SLURM_NPROCS" = "x" ]
then
    if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
    then
        SLURM_NTASKS_PER_NODE=1
    fi
    SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
    else
        if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
        then
            SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
        fi
fi

# ----- SET UP TEMPORARY ENVIRONMENT TO DO COMPUTATIONS IN ----- #
srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \" slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID > /tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

# ----- LOAD CONDA ENVIRONMENT ----- #
source ~/<path_to_conda>/conda.sh
conda activate <your_env>

# ----- LOAD GCC FOR GRAPH PARTITIONING ----- #
module load gcc

# ----- PERFORM GRAPH PARTITIONING ----- #
# CHECK IF SUBGRAPHS EXIST
SUBGRAPH_FILE=<path_to_code_folder>/code/data/subgraph_list
if [ -f "$SUBGRAPH_FILE" ]
then
  rm -r <path_to_code_folder>/code/data/subgraph_list
  rm -r <path_to_code_folder>/code/data/adj_matrices
  echo deleting old files...
fi

# MAKE NEW DIRECTORIES FOR SUBGRAPH INFORMATION
echo making new files ...
mkdir <path_to_code_folder>/code/data/subgraph_list
mkdir <path_to_code_folder>/code/data/adj_matrices

# PARTITION THE GRAPH WITH PYMETIS
python <path_to_code_folder>/code/create_coarse_graph.py --num_nodes=$SLURM_NPROCS

# ----- PARSE USER INPUTS ----- #
while getopts e:l: flag
do
    case "${flag}" in
        e) epochs=${OPTARG};;
        l) lr=${OPTARG};;
    esac
done

# ----- RUN COMMAND ON MULTIPLE NODES ----- #
mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS python <path_to_code_folder>/code/distributed_gcn_trainer.py --num_nodes=$SLURM_NPROCS --epochs=$epochs --lr=$lr

rm /tmp/hosts.$SLURM_JOB_ID
```

Once the `.sh` file has been included in the `code` folder, the following command can be ran in AiMOS to train a GCN with DGD:

```
sbatch -N<num_nodes> --ntasks-per-node=<num_gpus> --gres=gpu:<num_gpus> -t 5:00 -o <path_to_code_folder>/code/output_file_name.out <path_to_code_folder>/code/runScript.sh -e 350 -l 50
```
