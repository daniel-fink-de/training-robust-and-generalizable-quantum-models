from pennylane import numpy as np
import matplotlib.pyplot as plt
from .file import load_trainable_robustness_output, load_training_trainable_output
from .file import load_non_trainable_robustness_output, load_training_non_trainable_output
from .plot import base
from .circuit import lipschitz_bound_trainable_encoding, lipschitz_bound_non_trainable_encoding
from .analyse_robustness_trainable import noise_levels

lambdas_to_plot = [0.00, 0.20, 0.50]

def plot(trainable_robustness_output,
          trainable_training_output, 
          non_trainable_robustness_output, 
          training_non_trainable_output):
    lambdas = trainable_robustness_output.lambdas
    epsilons = trainable_robustness_output.epsilons
    accuracies_mean = np.mean(trainable_robustness_output.test_accuracies, axis=2)
    accuracies_min = np.min(trainable_robustness_output.test_accuracies, axis=2)

    # Collect only the values that we plot
    accuracies_mean_to_plot = []
    accuracies_min_to_plot = []
    lip_bounds_to_plot = []
    for i_lambda, lamb in enumerate(lambdas):
        # Check if we skip the lambda
        if lamb not in lambdas_to_plot:
            continue
        accuracies_mean_to_plot.append(accuracies_mean[i_lambda])
        accuracies_min_to_plot.append(accuracies_min[i_lambda])

        # BugFix: Do not use the "Lipschitz bound" in the robustness_output,
        # since it is actually the Lipschitz regularization, i.e.,
        # regularization = sum_i ||w_i * 0.5||^2 = sum_i sum_j |w_ij * 0.5|^2 but
        # bound = sum_i ||w_i * 0.5|| = sum_i sqrt( sum_j |w_ij * 0.5|^2 ),
        # where i is the index for the gate and j goes from 1-2 since it has the same
        # size as the data.
        # Instead, we here recalculate the true Lipschitz bound again from the weights
        
        # Get best weights as best epoch over all ensambles
        costs = trainable_training_output.cost_over_epochs[i_lambda]
        best_ensamble_idx, best_epoch_idx = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        weights = trainable_training_output.weights_over_epochs[i_lambda, best_ensamble_idx, best_epoch_idx]
        lip_bound = lipschitz_bound_trainable_encoding(weights)
        lip_bounds_to_plot.append(lip_bound)
    
    non_trainable_accuracies = np.min(non_trainable_robustness_output.test_accuracies, axis=2)[0]
    # Calculate the (fxied) Lipschitz bound for the non trainable encoding.
    non_trainable_encoding_weights = training_non_trainable_output.weights_over_epochs[0, 0, 0]
    non_trainable_lip_bound = lipschitz_bound_non_trainable_encoding(non_trainable_encoding_weights)
    lip_bounds_to_plot = lip_bounds_to_plot + [non_trainable_lip_bound]

    markers = ["*", ".", "d", "v"]
    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i_reg, reg in enumerate(lambdas_to_plot):
        ax.plot(epsilons,
                 accuracies_min_to_plot[i_reg], 
                 linewidth=1.0,
                   linestyle="dashed", 
                   marker=markers[i_reg], 
                   color=colors[i_reg], 
                   label=f"$\lambda = {reg}$")
    
    ax.plot(epsilons,
            non_trainable_accuracies,
            linewidth=1.0,
            linestyle="dashed",
            marker=markers[3], 
            color=colors[3], 
            label=f"Fixed")

    ax.set_xlabel("Noise Level $\\bar{\\varepsilon}$")
    ax.set_ylabel("Test Accuracy (worst-case)")
    y_tick_labels = ["0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"]
    y_tick_values = [float(y) for y in y_tick_labels]
    ax.set_yticks(y_tick_values, y_tick_labels)
    ax.set_ylim(min(y_tick_values), max(y_tick_values))
    x_tick_values = np.arange(0.0, 1.1, 0.1)
    x_tick_labels = [str(np.round(l, 1)) for l in x_tick_values]
    ax.set_xticks(x_tick_values, x_tick_labels)
    ax.set_xlim(min(noise_levels), max(noise_levels))

    ax.grid(axis="y")

    # Add the Lipschitz Bound Plot
    sub_ax = fig.add_axes([0.21, 0.195, 0.33, 0.26])
    lambda_str = [f"$\lambda = {l}$" for l in lambdas_to_plot] + ["Fixed"]
    bars = sub_ax.bar(lambda_str, lip_bounds_to_plot, color=colors, width=0.6)
    y_tick_values2 = [1e+0, 1e+1, 1e+2]
    y_tick_labels2 = ["$10^{0}$", "$10^{1}$", "$10^{2}$"]
    sub_ax.set_yscale("log")
    sub_ax.set_yticks(y_tick_values2, y_tick_labels2, fontsize=14)
    sub_ax.set_ylim(min(y_tick_values2), max(y_tick_values2))
    sub_ax.xaxis.set_visible(False)
    sub_ax.set_title("Lipschitz Bound", fontsize=14)

    for bar in bars:
        yval = round(bar.get_height(), 2)
        plt.text(bar.get_x() + 0.065, yval + yval * 0.14, yval, fontsize=10)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"./plots/robustness.pdf")
    return

def run():
    trainable_robustness_output = load_trainable_robustness_output()
    trainable_training_output = load_training_trainable_output()
    non_trainable_robustness_output = load_non_trainable_robustness_output()
    non_trainable_training_output = load_training_non_trainable_output()
    plot(trainable_robustness_output,
          trainable_training_output,
            non_trainable_robustness_output, 
            non_trainable_training_output)

    return

if __name__ == "__main__":
    run()