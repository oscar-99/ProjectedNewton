# Requirements

Our implementation is built in pytorch torch.func (https://pytorch.org/docs/stable/func.html). torch.func is a jax-like functional implementation of pytorch.

The project was developed using a conda environment with the following packages and dependencies
- python 3.9
- pytorch 2.0.1 + torchvision
- scikit-learn 1.2.2
- matplotlib 3.7.1


# Usage 

## Running the Experiments.
To run experiments execute the command "run_examples.py -e example". Where example is desired experiment. The examples available are 

- L1_log_digits: a small scale version of the MNIST. This should be runnable on a CPU.
- L1_log_MNIST: L1 regularised logistic regression on binarised MNIST.
- L1_multi_CIFAR: L1 regularised multinomial regression on the CIFAR10 dataset.
- L1_MLP_fashionMNIST: L1 regularised simple neural network on the fashionMNIST.
- NNMF_text_cosine: NNMF with the cosine loss on the 20newsgroupdataset.
- NNMF_image_noncvx: NNMF with frobenius loss and nonconvex regularisation on the faces dataset.

Each run will generate a new timestamped folder in the results folder. Within the results folder a .json file will be created for each algorithm. The "settings" key includes the parameter settings used to run each method. Our runs were performed on a GPU cluster.

## Generating Plots 
To view the plots for a result run the command "plot_examples.py -f folder". Where folder is the name of the folder the results are in. e.g. "./resuls/{folder}/". The plots used in the paper are included in the "plots" folder.

Extended results analysis is available in the results_analysis.py file. This file contains sections separated by "# %%" which can be run in a jupyter like environment in vscode or copy pasted into another folder to generate additional plots for each of the results, e.g., the local convergence plots used in the paper and the low rank representations generated by NNMF. 

## Raw Data
The raw data .jsons are too large for github, so to get raw data unzip the contents of results.zip into the results folder. 