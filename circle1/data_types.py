from dataclasses import dataclass

@dataclass
class TrainingOutput: # Can't rename this since the type is also serialized in pickle
    """ 
    A class for storing and serializing all training related metrics for the trainable QML model.
    """
    x_train: ...  # array of 2d points used for training
    y_train: ...  # array of labels used for training
    x_validation: ...  # array of 2d points used for validation
    y_validation: ...  # array of labels used for validation
    lambdas: ...  # array of regularization parameters lambda
    cost_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    train_accuracy_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    validation_accuracy_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    weights_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch, n_layer, n_qubit, x_dim, x_dim)
    biases_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch, n_layer, n_qubit, x_dim)
    lipschitz_bound_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch) # Can't rename to lip regularization since the name is also serialized in pickle

@dataclass
class TrainingNonTrainableOutput:
    """ 
    A class for storing and serializing all training related metrics for the non trainable QML model.
    """
    x_train: ...  # array of 2d points used for training
    y_train: ...  # array of labels used for training
    x_validation: ...  # array of 2d points used for validation
    y_validation: ...  # array of labels used for validation
    lambdas: ...  # array of regularization parameters lambda
    cost_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    train_accuracy_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    validation_accuracy_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch)
    weights_over_epochs: ...  # array of shape (n_lambda, n_ensamble, n_epoch, n_layer, n_qubit, 3)

@dataclass
class RobustnessOutput: # Can't rename this since the type is also serialized in pickle
    """ 
    A class for storing and serializing all robustness related metrics for the trainable encoding QML model.
    """
    lambdas: ...  # array of regularization parameters lambda
    epsilons: ...  # array of noise levels epsilon
    test_accuracies: ...  # array of shape (n_lambda, n_noise, n_ensamble)
    lipschitz_bounds: ...  # array of shape (n_lambda) # Can't rename to lip regularization since the name is also serialized in pickle
    x_test: ...  # 2d array for testing data points of shape (n_test, 2)
    y_test: ...  # 1d array for labels of shape (n_test)

@dataclass
class NonTrainableRobustnessOutput:
    """ 
    A class for storing and serializing all robustness related metrics for the non trainable encoding QML model.
    """
    lambdas: ...  # array of regularization parameters lambda
    epsilons: ...  # array of noise levels epsilon
    test_accuracies: ...  # array of shape (n_lambda, n_noise, n_ensamble)
    x_test: ...  # 2d array for testing data points of shape (n_test, 2)
    y_test: ...  # 1d array for labels of shape (n_test)

@dataclass
class TrainableGeneralizationOutput:
    """ 
    A class for storing and serializing all generalization related metrics for the trainable encoding QML model.
    """
    lambdas: ...  # array of regularization parameters lambda
    test_accuracies: ...  # array of shape (n_lambda)
    lipschitz_bounds: ...  # array of shape (n_lambda)
    x_test: ...  # 2d array for testing data points of shape (n_test, 2)
    y_test: ...  # 1d array for labels of shape (n_test)

@dataclass
class NonTrainableGeneralizationOutput:
    """ 
    A class for storing and serializing all generalization related metrics for the non trainable encoding QML model.
    """
    lambdas: ...  # array of regularization parameters lambda
    test_accuracies: ...  # array of shape (n_lambda)
    lipschitz_bounds: ...  # array of shape (n_lambda)
    x_test: ...  # 2d array for testing data points of shape (n_test, 2)
    y_test: ...  # 1d array for labels of shape (n_test)