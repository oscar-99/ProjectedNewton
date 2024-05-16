import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", help="The example to run.", nargs="+")
parser.add_argument("-s", "--save", help="Save location.")
args = parser.parse_args()

ex = args.examples

if ex is None:
    ex=[]


save_folder = args.save

print("Running examples {}.".format(ex))

if "L1_log_digits" in ex:
    from examples import L1_logistic_digits
    L1_logistic_digits.run(save_folder=save_folder)

if "L1_log_MNIST" in ex:
    from examples import L1_logistic_MNIST
    L1_logistic_MNIST.run(save_folder=save_folder)

if "L1_multi_CIFAR" in ex:
    from examples import L1_multinomial_CIFAR
    L1_multinomial_CIFAR.run(save_folder=save_folder)

if "L1_MLP_fashionMNIST" in ex:
    from examples import L1_MLP_fashionMNIST
    L1_MLP_fashionMNIST.run(save_folder=save_folder)

if "NNMF_text_cosine" in ex:
    from examples import NNMF_text_cosine
    NNMF_text_cosine.run(save_folder=save_folder)

if "NNMF_image_noncvx" in ex:
    from examples import NNMF_image_noncvx
    NNMF_image_noncvx.run(save_folder=save_folder)