import matplotlib.pyplot as plt
from pennylane import numpy as np
from .file import load_training_trainable_output
from .circuit import trainable_encoding_predict
from .plot import base

def plot_circle(x, y, fig=None, ax=None, radius=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == -1
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="bisque", s=1)
    ax.scatter(x[blues, 0], x[blues, 1], c="lightskyblue", s=1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-1, +1)
    ax.set_ylim(-1, +1)

    if radius is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = np.cos(theta) * radius
        y_circle = np.sin(theta) * radius
        ax.plot(x_circle, y_circle, color="dimgrey", linestyle="--")
    
    return

def plot_data(x_test, y_test, y_no_lamb, y_lamb, radius):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_circle(x_test, y_test, fig, axes[0], radius)
    plot_circle(x_test, y_no_lamb, fig, axes[1], radius)
    plot_circle(x_test, y_lamb, fig, axes[2], radius)
    axes[0].set_title("Ground Truth")
    axes[1].set_title("$\lambda = 0.0$")
    axes[2].set_title("$\lambda = 0.2$")
    plt.tight_layout()
    plt.savefig(f"./plots/prediction.pdf")
    return

def run():
    # The decision boundary
    radius = np.sqrt(2 / np.pi)

    # Make a regular grid
    resolution = 0.01
    x = np.arange(-1, 1+resolution, resolution)
    y = np.arange(-1, 1+resolution, resolution)
    X,Y = np.meshgrid(x,y)
    x_test = np.vstack((X.flatten(), Y.flatten())).T

    # Classify accordingly
    y_test = []
    for x in x_test:
        y = -1
        if np.linalg.norm(x - [0.0, 0.0]) < radius:
            y = 1
        y_test.append(y)
    y_test = np.array(y_test)

    # Get best weights as best epoch over all ensambles
    output = load_training_trainable_output()
    costs = output.cost_over_epochs

    no_lamb_idx = 0
    lamb_idx = 5

    no_lamb_best_ensamble_idx, no_lamb_best_epoch_idx = np.unravel_index(np.argmin(costs[no_lamb_idx], axis=None), costs[no_lamb_idx].shape)
    lamb_best_ensamble_idx, lamb_best_epoch_idx = np.unravel_index(np.argmin(costs[lamb_idx], axis=None), costs[lamb_idx].shape)
    
    no_lamb_best_weights = output.weights_over_epochs[no_lamb_idx, no_lamb_best_ensamble_idx, no_lamb_best_epoch_idx]
    no_lamb_best_biases = output.biases_over_epochs[no_lamb_idx, no_lamb_best_ensamble_idx, no_lamb_best_epoch_idx]

    lamb_best_weights = output.weights_over_epochs[lamb_idx, lamb_best_ensamble_idx, lamb_best_epoch_idx]
    lamb_best_biases = output.biases_over_epochs[lamb_idx, lamb_best_ensamble_idx, lamb_best_epoch_idx]

    y_no_lamb = trainable_encoding_predict(no_lamb_best_weights, no_lamb_best_biases, x_test)
    y_lamb = trainable_encoding_predict(lamb_best_weights, lamb_best_biases, x_test)
    plot_data(x_test, y_test, y_no_lamb, y_lamb, radius)

    return

if __name__ == "__main__":
    run()