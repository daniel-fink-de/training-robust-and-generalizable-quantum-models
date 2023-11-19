from pennylane import numpy as np
import dask
import time
from dask.distributed import LocalCluster, Client
from .data import circle
from .file import load_training_trainable_output, save_trainable_generalization_output
from .circuit import trainable_encoding_predict, calculate_accuracy, lipschitz_bound_trainable_encoding
from .data_types import TrainableGeneralizationOutput

# Experiment configuration
n_test = 10_000 # How many points to evaluate

def calculate_test_accuracy_trainable_encoding(x_test, y_test, weights, biases, lamb):
    """
    Calculate the test accuracy for the trainable encoding given weights and biases for a test set.

    Args:
        x_test (array[float]): 2-d array of input vectors of shape (n_testm, 2)
        y_test (array[float]): 1-d array of targets of shape (n_test)
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)
        biases (array[float]): array of biases of shape (n_layer, n_qubit, x_dim)
        lamb (float): regularization parameter

    Returns:
        array[float]: 1-d array of test accuracies of shape (n_ensamble)
    """
    print(f"Analyse generalization for lambda {lamb}")
    predictions = trainable_encoding_predict(weights, biases, x_test)
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

    training_output = load_training_trainable_output()
    lambdas = training_output.lambdas
    n_lambda = len(lambdas)
    lip_bounds = []

    jobs = []
    for i_lamb, lamb in enumerate(lambdas):
        # Get best weights as best epoch over all ensambles
        costs = training_output.cost_over_epochs[i_lamb]
        best_ensamble_idx, best_epoch_idx = np.unravel_index(np.argmin(costs, axis=None), costs.shape)

        weights_best = training_output.weights_over_epochs[i_lamb, best_ensamble_idx, best_epoch_idx]
        biases_best = training_output.biases_over_epochs[i_lamb, best_ensamble_idx, best_epoch_idx]

        # BugFix: Do not use the "Lipschitz bound" in the training_output,
        # since it is actually the Lipschitz regularization, i.e.,
        # regularization = sum_i ||w_i * 0.5||^2 = sum_i sum_j |w_ij * 0.5|^2 but
        # bound = sum_i ||w_i * 0.5|| = sum_i sqrt( sum_j |w_ij * 0.5|^2 ),
        # where i is the index for the gate and j goes from 1-2 since it has the same
        # size as the data.
        # Instead, we here recalculate the true Lipschitz bound again from the weights
        
        # Get best weights as best epoch over all ensambles
        weights = training_output.weights_over_epochs[i_lamb, best_ensamble_idx, best_epoch_idx]
        lip_bound = lipschitz_bound_trainable_encoding(weights)
        lip_bounds.append(lip_bound)

        job = dask.delayed(calculate_test_accuracy_trainable_encoding)(x_test, y_test, weights_best, biases_best, lamb)
        jobs.append(job)
    
    results = dask.compute(*jobs)
    accuracies = np.zeros(shape=(n_lambda))
    lip_bounds = np.array(lip_bounds, dtype=float)

    for i_lamb in range(n_lambda):
        accuracies[i_lamb] = results[i_lamb]

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total generalization analysis time: {int(round(total_time, 0))} seconds ({round(total_time/60, 2)} minutes)")

    output = TrainableGeneralizationOutput(lambdas,
                                  accuracies,
                                  lip_bounds,
                                  x_test,
                                  y_test)
    
    save_trainable_generalization_output(output)

    return

if __name__ == "__main__":
    run()