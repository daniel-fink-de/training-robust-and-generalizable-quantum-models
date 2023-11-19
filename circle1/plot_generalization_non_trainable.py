from pennylane import numpy as np
import matplotlib.pyplot as plt
from .file import load_non_trainable_generalization_output
from .plot import base
from .training_non_trainable import lambdas

lambdas_non_trainable_to_plot = lambdas

def plot(generalization_non_trainable_output):
    # Plot generalization for non trainable encoding
    lambdas = generalization_non_trainable_output.lambdas
    test_accs = generalization_non_trainable_output.test_accuracies
    lip_bounds = generalization_non_trainable_output.lipschitz_bounds

    # Collect only the values that we plot
    test_accs_to_plot = []
    lip_bounds_to_plot = []
    for i_lambda, lamb in enumerate(lambdas):
        # Check if we skip the lambda
        if lamb not in lambdas_non_trainable_to_plot:
            continue
        test_accs_to_plot.append(test_accs[i_lambda])
        lip_bounds_to_plot.append(lip_bounds[i_lambda])

    # Plot the Test Accuracy over Lambdas
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("Regularization parameter $\lambda$")
    ax.set_ylabel("Accuracy", color="tab:blue")

    y_tick_values = np.round(np.arange(0.76, 1.01, 0.04), 2)
    y_tick_labels = [format(l, f'.{2}f') for l in y_tick_values]
    ax.set_yticks(y_tick_values, y_tick_labels, color="tab:blue")
    ax.set_ylim(min(y_tick_values), max(y_tick_values))

    x_tick_labels = ["$0$", 
                       "$10^{-5}$",
                       "$5 \cdot 10^{-5}$",
                       "$10^{-4}$",
                       "$5 \cdot 10^{-4}$",
                       "$10^{-3}$",
                       "$5 \cdot 10^{-3}$",
                       "$10^{-2}$",
                       "$5 \cdot 10^{-2}$",
                       "$10^{-1}$"]
    x_tick_values = range(len(lambdas_non_trainable_to_plot))
    #x_tick_labels = [format(l, f'.{2}f') for l in lambdas_non_trainable_to_plot]
    ax.set_xticks(x_tick_values, x_tick_labels, fontsize=14)
    ax.set_xlim(min(x_tick_values), max(x_tick_values))
    ax.grid(axis="y")

    ax.plot(x_tick_values, test_accs_to_plot, linewidth=1.0, linestyle="dashed", marker="*", color="tab:blue", label="Test Accuracy")

    # Also plot the Lipschitz Bound over Lambdas
    ax2 = ax.twinx()
    ax2.plot(x_tick_values, lip_bounds_to_plot, linewidth=1.0, linestyle="dashed", marker=".", color="tab:red", label="Lipschitz Bound")
    y_tick_values2 = range(54, 61, 1)
    y_tick_labels2 = [str(l) for l in y_tick_values2]
    ax2.set_ylabel("Bound", color="tab:red")
    ax2.set_yticks(y_tick_values2, y_tick_labels2, color="tab:red")
    ax2.set_ylim(min(y_tick_values2), max(y_tick_values2))

    # Show legends for both curves
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.savefig(f"./plots/generalization_non_trainable.pdf")
    return

def run():
    generalization_non_trainable_output = load_non_trainable_generalization_output()
    plot(generalization_non_trainable_output)

    return

if __name__ == "__main__":
    run()