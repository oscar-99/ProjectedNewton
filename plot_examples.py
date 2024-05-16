import argparse
import os
import utils.utils as utils
from utils.utils_plot import plot_example, save_plot
import matplotlib as mpl

mpl.rcParams.update({'font.size': 12, 'font.weight' : "bold"})


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", help="The example to run.", nargs="+")
parser.add_argument("--all", help="Plots all of the included examples", action="store_true")
parser.add_argument("--time", action="store_true", help="Plot results by time instead of by oracle calls")
args = parser.parse_args()

ex = args.examples

if ex is None:
    ex=[]

if args.all:
    ex = ["L1_logistic_digits", "L1_logistic_MNIST", "L1_multinomial_CIFAR10", "L1_MLP_fashion", "NNMF_image_noncvx", "NNMF_text_cosine"]
if args.time:
    by_time = True
else:
    by_time = False

print("Plotting examples {}.".format(ex))

for e in ex:
    if "NNMF" in e:
        ops_weighting = (1, 1, 2)
    if "L1" in e:
        ops_weighting = (1, 1, 4)
    
    if by_time:
        plot = plot_example(e, by_time=True,  ops_weighting=ops_weighting)
        name = e + "_time"
    else:
        plot = plot_example(e, ops_weighting=ops_weighting)
        name = e

    save_plot(plot, name)