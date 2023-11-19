import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from .plot import base
from .circuit import trainable_encoding, non_trainable_encoding
from .training_trainable import n_layer as n_layer_trainable
from .training_trainable import n_qubit as n_qubit_trainable
from .training_non_trainable import n_layer as n_layer_non_trainable
from .training_non_trainable import n_qubit as n_qubit_non_trainable

def plot_trainable_encoding():
    x_dim = 2
    weights = np.zeros(shape=(n_layer_trainable, n_qubit_trainable, x_dim, x_dim), requires_grad=False)
    biases = np.zeros(shape=(n_layer_trainable, n_qubit_trainable, x_dim), requires_grad=False)
    x = np.zeros(shape=2, requires_grad=False)
    fig, ax = qml.draw_mpl(trainable_encoding)(weights, biases, x)
    print(f"Trainable encoding features {weights.size} weights and {biases.size} biases, i.e., {weights.size + biases.size} total.")
    plt.savefig("./plots/trainable_encoding.pdf")
    return

def plot_non_trainable_encoding():
    weights = np.zeros(shape=(n_layer_non_trainable, n_qubit_non_trainable, 3), requires_grad=False)
    x = np.zeros(shape=2, requires_grad=False)
    fig, ax = qml.draw_mpl(non_trainable_encoding)(weights, x)
    print(f"Non-Trainable encoding features {weights.size} weights.")
    plt.savefig("./plots/non_trainable_encoding.pdf")
    return

def run():
    plot_trainable_encoding()
    plot_non_trainable_encoding()
    return

if __name__ == "__main__":
    run()