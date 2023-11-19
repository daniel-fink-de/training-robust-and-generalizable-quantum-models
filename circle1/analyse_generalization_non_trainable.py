from pennylane import numpy as np
import dask
import time
from dask.distributed import LocalCluster, Client
from .data import circle
from .file import load_training_non_trainable_output, save_non_trainable_generalization_output
from .circuit import non_trainable_encoding_predict, calculate_accuracy, lipschitz_bound_non_trainable_encoding
from .data_types import NonTrainableGeneralizationOutput

# Experiment configuration
n_test = 10_000 # How many points to evaluate

def calculate_test_accuracy(x_test, y_test, weights, lamb):
    """
    Calculate the test accuracy for the non trainable encoding given weights for a test set.

    Args:
        x_test (array[float]): 2-d array of input vectors of shape (n_testm, 2)
        y_test (array[float]): 1-d array of targets of shape (n_test)
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)
        lamb (float): regularization parameter

    Returns:
        array[float]: 1-d array of test accuracies of shape (n_ensamble)
    """
    print(f"Analyse generalization for lambda {lamb}")
    predictions = non_trainable_encoding_predict(weights, x_test)
    accuracy = calculate_accuracy(y_test, predictions)
    return accuracy

def run():
    start_time = time.time()

    # Setup parallelization
    dask.config.set({'logging.distributed': 'error'})
    cluster = LocalCluster(n_workers=6, threads_per_worker=1, processes=True)  # Launches a scheduler and workers locally
    cluster.scale(6)   
    client = Client(cluster)  # Connect to distributed cluster and override default
    print(f"Dask Dashboard: {client.dashboard_link}")

    # Create test set to collect statistics
    x_test, y_test = circle(n_test)

    # Scale data from [-1, +1] to [-pi, +pi] in order to
    # fully access all degrees of freedom in the rotation gates
    x_test = np.multiply(x_test, np.pi)

    training_output = load_training_non_trainable_output()
    lambdas = training_output.lambdas
    n_lambda = len(lambdas)
    lip_bounds = []

    jobs = []
    for i_lamb, lamb in enumerate(lambdas):
        # Get best weights as best epoch over all ensambles
        costs = training_output.cost_over_epochs[i_lamb]
        best_ensamble_idx, best_epoch_idx = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        weights_best = training_output.weights_over_epochs[i_lamb, best_ensamble_idx, best_epoch_idx]
        lip_bound = lipschitz_bound_non_trainable_encoding(weights_best)
        lip_bounds.append(lip_bound)

        job = dask.delayed(calculate_test_accuracy)(x_test, y_test, weights_best, lamb)
        jobs.append(job)
    
    results = dask.compute(*jobs)
    accuracies = np.zeros(shape=(n_lambda))

    for i_lamb in range(n_lambda):
        accuracies[i_lamb] = results[i_lamb]

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total generalization analysis time: {int(round(total_time, 0))} seconds ({round(total_time/60, 2)} minutes)")

    # Rescale data from [-pi, +pi] to [-1, +1]
    x_test = np.divide(x_test, np.pi)

    output = NonTrainableGeneralizationOutput(lambdas,
                                  accuracies,
                                  lip_bounds,
                                  x_test,
                                  y_test)
    
    save_non_trainable_generalization_output(output)

    return

if __name__ == "__main__":
    run()