# MPI-Neural-Network

Welcome to my MPI implementation of a simple Multilayer Perceptron (MLP) for the "Concurrent and Parallel Programming" subject at my Computational Mathematics university degree.

This program is designed to practice basic Deep Learning concepts and their intersection with efficient computation. It should not be used in production or research, as it is an exercise and has several limitations.

The program uses MPI to handle a specific number of neurons per process, avoiding the need to share and replicate the weights in every process. However, it is still necessary for the processes to share the intermediate activations, the output, and the inputs to the network.

## Usage

To use the program:
1. Compile it using the command: `mpicc neural_network.c -o neural_network -lm`
2. Run it using the command: `mpirun -np <NPROCS> ./neural_network tiny_nn.txt`

The configuration file passed as an argument should have the following structure:
- Number of inputs
- Inputs
- Number of layers
- Number of neurons per layer
- Number of outputs
- Desired outputs
You can find examples in this repository such as `tiny_nn.txt`, `small_nn.txt`, or `big_nn.txt`.

## Limitations

Please note the following limitations:
- Currently, only forward pass is implemented.
- All the hidden layers of the MLP have the same size. Only input and output layers can be different.
- Only Relu and Tanh activation functions are implemented.
- The inputs should be normalized with mean 0 and standard deviation 1 (for numerical stability).
- The number of neurons per layer MUST be a multiple of the number of processes.

Thank you for using my program and I hope it helps you in your understanding of concurrent and parallel programming with MLPs.
