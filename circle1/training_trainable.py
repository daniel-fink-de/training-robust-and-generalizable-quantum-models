from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import dask
import time
from dask.distributed import LocalCluster, Client
from .file import save_training_trainable_output
from .data import circle
from .circuit import trainable_encoding, lipschitz_regularization, trainable_encoding_predict, calculate_accuracy
from .data_types import TrainingOutput

# Experiment configuration
n_ensamble = 9  # Number of initial weights
n_qubit = 3
n_layer = 3
n_epoch = 200
learning_rate = 0.1
lambdas = np.array([0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], dtype=float, requires_grad=False)
n_training_samples = 200
n_validation_samples = 1000

def cost(weights, biases, x, y, lamb):
    """
    The cost function to be minimized. We get outputs y^hat in [-1,+1] from the PQC and
    apply the sig function for predictions (not for the cost function in training).

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)
        biases (array[float]): array of biases of shape (n_layer, n_qubit, x_dim)
        x (array[float]): 2-d array of input vectors
        y (array[float]): 1-d array of targets

    Returns:
        float: loss value to be minimized (MSE loss)
    """
    loss = 0.0
    for i in range(len(x)):
        expect = trainable_encoding(weights, biases, x[i])
        loss = loss + (expect - y[i]) ** 2

    loss = loss / len(x)
    lip_reg = lipschitz_regularization(weights)
    loss = loss + lamb * lip_reg
    return loss

def train(lamb, weights, biases, x_train, y_train, x_validation, y_validation, lr, n_epoch):
    start_time = time.time()

    weights = weights.copy()
    biases = biases.copy()

    x_dim = x_train[0].shape[0]

    # Define metrics to store
    cost_over_epochs = np.zeros(n_epoch, requires_grad=False)
    train_accuracy_over_epochs = np.zeros(n_epoch, requires_grad=False)
    validation_accuracy_over_epochs = np.zeros(n_epoch, requires_grad=False)
    weights_over_epochs = np.zeros(shape=(n_epoch, n_layer, n_qubit, x_dim, x_dim), requires_grad=False)
    biases_over_epochs = np.zeros(shape=(n_epoch, n_layer, n_qubit, x_dim), requires_grad=False)
    lipschitz_regularization_over_epochs = np.zeros(n_epoch, requires_grad=False)

    # Train using Adam optimizer
    opt = AdamOptimizer(lr, beta1=0.9, beta2=0.999)

    # Predict train data
    predictions_train = trainable_encoding_predict(weights, biases, x_train)
    accuracy_train = calculate_accuracy(y_train, predictions_train)

    # Predict validation data
    predictions_validation = trainable_encoding_predict(weights, biases, x_validation)
    accuracy_validation = calculate_accuracy(y_validation, predictions_validation)

    # Calculate Lipschitz Bound
    lip_reg = lipschitz_regularization(weights)

    loss = cost(weights, biases, x_train, y_train, lamb)
    cost_over_epochs[0] = loss
    train_accuracy_over_epochs[0] = accuracy_train
    validation_accuracy_over_epochs[0] = accuracy_validation
    weights_over_epochs[0] = weights.copy()
    biases_over_epochs[0] = biases.copy()
    lipschitz_regularization_over_epochs[0] = lip_reg

    print(
        "Lambda: {:2f} | Epoch: {:2d} | Cost: {:3f} | Train accuracy: {:3f} | Validation Accuracy: {:3f}".format(
            lamb, 0, loss, accuracy_train, accuracy_validation
        )
    )

    # Run the optimization
    for it in range(1,n_epoch):
        weights, biases, _, _, _ = opt.step(cost, weights, biases, x_train, y_train, lamb)

        # Predict train data
        predictions_train = trainable_encoding_predict(weights, biases, x_train)
        accuracy_train = calculate_accuracy(y_train, predictions_train)
        loss = cost(weights, biases, x_train, y_train, lamb)

        # Predict validation data
        predictions_validation = trainable_encoding_predict(weights, biases, x_validation)
        accuracy_validation = calculate_accuracy(y_validation, predictions_validation)

        # Calculate Lipschitz Bound
        lip_reg = lipschitz_regularization(weights)

        cost_over_epochs[it] = loss
        train_accuracy_over_epochs[it] = accuracy_train
        validation_accuracy_over_epochs[it] = accuracy_validation
        weights_over_epochs[it] = weights.copy()
        biases_over_epochs[it] = biases.copy()
        lipschitz_regularization_over_epochs[it] = lip_reg

        res = [lamb, it, loss, accuracy_train, accuracy_validation]
        print(
            "Lambda: {:2f} | Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Validation accuracy: {:3f}".format(
                *res
            )
        )
    
    end_time = time.time()
    single_time = end_time - start_time
    print(f"Single training time: {int(round(single_time, 0))} seconds ({round(single_time/60, 2)} minutes)")

    return cost_over_epochs, train_accuracy_over_epochs, validation_accuracy_over_epochs, weights_over_epochs, biases_over_epochs, lipschitz_regularization_over_epochs

def run():
    # Setup parallelization
    dask.config.set({'logging.distributed': 'error'})
    cluster = LocalCluster(n_workers=6, threads_per_worker=1, processes=True)  # Launches a scheduler and workers locally
    cluster.scale(6)   
    client = Client(cluster)  # Connect to distributed cluster and override default
    print(f"Dask Dashboard: {client.dashboard_link}")

    # Load data
    x_train, y_train = circle(n_training_samples)
    x_validation, y_validation = circle(n_validation_samples)

    # Derive parameters
    n_lambda = len(lambdas)

    # Derive experiment constants
    x_dim = x_train[0].shape[0]

    # Initialize random parameters
    weights = np.random.uniform(low=0.0, high=np.pi, size=(n_ensamble, n_layer, n_qubit, x_dim, x_dim), requires_grad=True)
    biases = np.random.uniform(low=0.0, high=2*np.pi, size=(n_ensamble, n_layer, n_qubit, x_dim), requires_grad=True)

    # Run the training in parallel
    # Check if we waste ressources (machine dependent)
    if n_lambda * n_ensamble % 6 != 0:
        print("WARNING: WE ARE WAISTING RESSOURCES")
        print(f"Number of jobs is {n_lambda * n_ensamble} but it should be divisible by 6.")
    start_time = time.time()
    jobs = [] # n_lambda * n_ensamble
    for i_lamb, lamb in enumerate(lambdas):
        for i_ensamble in range(n_ensamble):
            job_index = i_lamb * (n_ensamble) + i_ensamble
            job = dask.delayed(train)(lamb, weights[i_ensamble], biases[i_ensamble], x_train, y_train, x_validation, y_validation, learning_rate, n_epoch)
            jobs.append(job)
    results = dask.compute(*jobs)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {int(round(total_time, 0))} seconds ({round(total_time/60, 2)} minutes)")

    # Define metrics to store
    cost_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch), requires_grad=False)
    train_accuracy_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch), requires_grad=False)
    validation_accuracy_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch), requires_grad=False)
    weights_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch, n_layer, n_qubit, x_dim, x_dim), requires_grad=False)
    biases_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch, n_layer, n_qubit, x_dim), requires_grad=False)
    lipschitz_regularization_over_epochs = np.zeros(shape=(n_lambda, n_ensamble, n_epoch), requires_grad=False)

    # Collect the results and put them into stored data types
    for i_lamb in range(n_lambda):
        for i_ensamble in range(n_ensamble):
            # Get the index to the job result
            exp_idx = i_lamb * (n_ensamble) + i_ensamble
            result = results[exp_idx]

            # Store the data
            cost_over_epochs[i_lamb, i_ensamble] = result[0]
            train_accuracy_over_epochs[i_lamb, i_ensamble] = result[1]
            validation_accuracy_over_epochs[i_lamb, i_ensamble] = result[2]
            weights_over_epochs[i_lamb, i_ensamble] = result[3]
            biases_over_epochs[i_lamb, i_ensamble] = result[4]
            lipschitz_regularization_over_epochs[i_lamb, i_ensamble] = result[5]

    output = TrainingOutput(x_train, 
                            y_train, 
                            x_validation, 
                            y_validation,
                            lambdas, 
                            cost_over_epochs, 
                            train_accuracy_over_epochs,
                            validation_accuracy_over_epochs,
                            weights_over_epochs,
                            biases_over_epochs,
                            lipschitz_regularization_over_epochs)
    
    save_training_trainable_output(output)

    return

if __name__ == "__main__":
    run()