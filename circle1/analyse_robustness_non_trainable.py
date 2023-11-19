from pennylane import numpy as np
import dask
import time
from dask.distributed import LocalCluster, Client
from .data import circle
from .file import load_training_non_trainable_output, save_non_trainable_robustness_output
from .circuit import non_trainable_encoding_predict, calculate_accuracy
from .data_types import NonTrainableRobustnessOutput
from .training_non_trainable import lambdas

# Experiment configuration
noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0]
lambdas_to_test = [0.0]
n_ensamble = 200 # To collect statistics
n_test = 1000 # How many points to evaluate

def calculate_accuracies(x_test, y_test, weights, noise_level, reg, eps):
    x_test = x_test.copy()
    print(f"Analyse robustness for level {noise_level} and lambda {reg}")
    accuracies = np.zeros(n_ensamble, dtype=float)
    for i_ensamble in range(n_ensamble):
        # Take the eps for the current ensamble
        current_eps = eps[i_ensamble]
        # Scale it linearily according to the noise level
        current_eps = np.multiply(current_eps, noise_level)
        x_disturbed = np.add(x_test, current_eps)

        # Scale data from [-1, +1] to [-pi, +pi] in order to
        # fully access all degrees of freedom in the rotation gates
        x_disturbed = np.multiply(x_disturbed, np.pi)

        # Perform the predictions
        predictions = non_trainable_encoding_predict(weights, x_disturbed)
        accuracy = calculate_accuracy(y_test, predictions)
        accuracies[i_ensamble] = accuracy
    return accuracies

def run():
    start_time = time.time()

    # Setup parallelization
    dask.config.set({'logging.distributed': 'error'})
    cluster = LocalCluster(n_workers=6, threads_per_worker=1, processes=True)  # Launches a scheduler and workers locally
    cluster.scale(6)   
    client = Client(cluster)  # Connect to distributed cluster and override default
    print(f"Dask Dashboard: {client.dashboard_link}")

    # Draw n_test random samples that serve as test data set
    x_test, y_test = circle(n_test)

    # Draw n_ensamble noise samples, that serve as the noise for noise level 1.0,
    # later, we scale this noise with the noise level
    eps = np.random.uniform(-1.0, +1.0, size=(n_ensamble, n_test, 2))

    # Load training output
    training_output = load_training_non_trainable_output()
    lambdas = training_output.lambdas
    n_lambda = len(training_output.lambdas)
    n_lambdas_to_analyse = len(lambdas_to_test)
    n_noise = len(noise_levels)

    jobs = []
    for i_reg in range(n_lambda):

        # Check if we shall analysze this lambda
        current_lambda = lambdas[i_reg]
        if current_lambda not in lambdas_to_test:
            continue

        # Get best weights as best epoch over all ensambles
        costs = training_output.cost_over_epochs[i_reg]
        best_ensamble_idx, best_epoch_idx = np.unravel_index(np.argmin(costs, axis=None), costs.shape)

        weights_best = training_output.weights_over_epochs[i_reg, best_ensamble_idx, best_epoch_idx]

        for i_noise, noise_level in enumerate(noise_levels):
            job_index = i_reg * (n_noise) + i_noise
            job = dask.delayed(calculate_accuracies)(x_test, y_test, weights_best, noise_level, current_lambda, eps)
            jobs.append(job)
    
    results = dask.compute(*jobs)
    accuracies = np.zeros((n_lambdas_to_analyse, n_noise, n_ensamble))

    for i_reg in range(n_lambdas_to_analyse):
        for i_noise, noise_level in enumerate(noise_levels):
            job_index = i_reg * (n_noise) + i_noise
            accuracies[i_reg, i_noise, :] = results[job_index]

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total robustness analysis time: {int(round(total_time, 0))} seconds ({round(total_time/60, 2)} minutes)")

    output = NonTrainableRobustnessOutput(lambdas_to_test,
                              noise_levels,
                              accuracies,
                              x_test,
                              y_test)
    
    save_non_trainable_robustness_output(output)

    return

if __name__ == "__main__":
    run()