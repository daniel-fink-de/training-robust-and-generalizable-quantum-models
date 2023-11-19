import pennylane as qml
from pennylane import numpy as np

n_qubit=3
dev = qml.device("lightning.qubit", wires=n_qubit)

@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def trainable_encoding(weights, biases, x):
    """
    A variational quantum circuit representing the trainable encoding QML model. We use a similar PQC as
    in the original Data Reuploading paper: https://quantum-journal.org/papers/q-2020-02-06-226/

    By contrast, however, per degree of freedom of the SU(2) gate, we put the full feature vector x inside, i.e.,

    U(x^T theta + p1, x^T phi + p2, x^T omega + p3)
    
    where theta, phi and omega are of the same dimension as the feature vector and p1, p2, p3 are scalars.

    The original paper simply does

    U(x_1 * theta_1 + phi_1, x_2 * theta_2 + phi_2, x_3 * theta_3 + phi_3), i.e., 

    each feature is encoded separately in one dimension with two trainable parameters

    An even simpler version is repeated sequential data encoding, i.e.,

    U(x_1, x_2, x_3) U(theta_1, theta_2, theta_3) U(x_1, x_2, x_3) U(theta_1, theta_2, theta_3),

    which is what the Furier Analysis papers are considering.

    Additionals:
    - If we have less dimensions than 3, we simply omitt the others
    - If we have more dimensions than 3, we add another SU(2) gate with the new dimensions

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)
        biases (array[float]): array of biases of shape (n_layer, n_qubit, x_dim)
        x (array[float]): single input vector of shape x_dim
        y (array[float]): single output scalar for tensor Pauli Z excpectation value

    Returns:
        float: expectation value of tensor Z, i.e., tr(rho(x,theta)O) with O = Z o Z o ... o Z.
    """
    n_layer, n_qubit, x_dim, _ = weights.shape
    for l in range(n_layer):
        # Trainable Layer
        for q in range(n_qubit):
            # We fill 3 dims and then check, if we need to add 1 or 2 0s for the omitted dimensions
            remaining_dimensions = x_dim % 3
            multiple_of_3_dims = x_dim - remaining_dimensions
            for i in range(0, multiple_of_3_dims, 3):
                dim1 = np.dot(weights[l,q,i + 0],x) + biases[l,q,i + 0]
                dim2 = np.dot(weights[l,q,i + 1],x) + biases[l,q,i + 1]
                dim3 = np.dot(weights[l,q,i + 2],x) + biases[l,q,i + 2]
                qml.Rot(dim1, dim2, dim3, wires=q)
            
            if remaining_dimensions == 1:
                dim1 = np.dot(weights[l,q,multiple_of_3_dims + 0],x) + biases[l,q,multiple_of_3_dims + 0]
                dim2 = 0.0
                dim3 = 0.0
                qml.Rot(dim1, dim2, dim3, wires=q)
            elif remaining_dimensions == 2:
                dim1 = np.dot(weights[l,q,multiple_of_3_dims + 0],x) + biases[l,q,multiple_of_3_dims + 0]
                dim2 = np.dot(weights[l,q,multiple_of_3_dims + 1],x) + biases[l,q,multiple_of_3_dims + 1]
                dim3 = 0.0
                qml.Rot(dim1, dim2, dim3, wires=q)

        # Entangling Layer
        if n_qubit >= 2:
            for q in range(n_qubit-1):
                qml.CNOT(wires=[q, q+1])
        if n_qubit >= 3:
            qml.CNOT(wires=[n_qubit-1,0])
        
    observable = qml.PauliZ(0)
    for i in range(1, n_qubit):
        observable = observable @ qml.PauliZ(i)

    return qml.expval(observable) # corresponds to <psi|Z o Z o ... o Z|psi>

@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def non_trainable_encoding(weights, x):
    """
    A variational quantum circuit representing the non-trainable encoding QML model. We use a similar PQC as
    in the partial Fourier paper: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032430

    That is, per qubit we apply a general SU(2) operation and encode each dimension of the data into a
    different degree of freedom in the gate, followed by the very same SU(2) gate but with trainable
    parameters, i.e.,

    U(x_1, x_2, x_3) -- U(theta_1, theta_2, theta_3),

    which is then followed by nearest neighbour CNOTs.

    Such a layer is repeated n_layer times.

    Additionals:
    - For the data encoding:
        - If we have less dimensions than 3, we simply omitt the others
        - If we have more dimensions than 3, we add another SU(2) gate with the new dimensions
    - For the trainable gates:
        - We always only apply one SU(2) operation with 3 trainable degrees of freedom per qubit per layer

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, 3)
        x (array[float]): single input vector of shape x_dim
        y (array[float]): single output scalar for tensor Pauli Z excpectation value

    Returns:
        float: expectation value of tensor Z, i.e., tr(rho(x,theta)O) with O = Z o Z o ... o Z.
    """
    n_layer, n_qubit, _ = weights.shape
    x_dim = len(x)
    for l in range(n_layer):

        # Data Unit
        for q in range(n_qubit):
            # We fill 3 dims and then check, if we need to add 1 or 2 0s for the omitted dimensions
            remaining_dimensions = x_dim % 3
            multiple_of_3_dims = x_dim - remaining_dimensions
            for i in range(0, multiple_of_3_dims, 3):
                qml.Rot(x[0], x[1], x[2], wires=q)
            if remaining_dimensions == 1:
                qml.Rot(x[0], 0.0, 0.0, wires=q)
            elif remaining_dimensions == 2:
                qml.Rot(x[0], x[1], 0.0, wires=q)

        # Trainable Unit
        for q in range(n_qubit):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)

        # Entangling Unit
        if n_qubit >= 2:
            for q in range(n_qubit-1):
                qml.CNOT(wires=[q, q+1])
        if n_qubit >= 3:
            qml.CNOT(wires=[n_qubit-1,0])
        
    observable = qml.PauliZ(0)
    for i in range(1, n_qubit):
        observable = observable @ qml.PauliZ(i)

    return qml.expval(observable) # corresponds to <psi|Z o Z o ... o Z|psi>

def lipschitz_regularization(weights):
    """
    Calculate the Lipschitz bound regularization given the weights, which is

    sum_i ||weight_i||^2 ||H_i||^2,

    for a sum over all weights, with H_i being the generator of the gate.
    
    Since we use Pauli Rotation gates (i.e. consider Gate decomposition of U3), we have
    H_i = 0.5 {X,Y,Z} and since ||H||=1 we only sum up the weights divided by 2 squared.

    https://arxiv.org/abs/2303.00618

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)

    Returns:
        float: Lipschitz regularization value
    """
    lipschitz_regularization = np.sum(np.square(np.multiply(weights, 0.5)))
    return lipschitz_regularization

def lipschitz_bound_trainable_encoding(weights):
    """
    Calculate the Lipschitz bound for the trainable enoding given the weights:

    L = 2 ||M|| sum_i ||w_i|| ||H_i||

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)

    Returns:
        float: Lipschitz bound of the trainable encoding.
    """
    lip_bound = np.multiply(weights, 0.5)
    lip_bound = np.square(lip_bound)
    lip_bound = np.sum(lip_bound, axis=3)
    lip_bound = np.sqrt(lip_bound)
    lip_bound = np.sum(lip_bound)
    lip_bound = lip_bound * 2.0
    return lip_bound

def lipschitz_bound_non_trainable_encoding(weights):
    """
    Calculate the Lipschitz bound for the non trainable enoding given the weights:

    L = 2 ||M|| sum_i ||w_i|| ||H_i|| = 2 ||M|| sum_i sqrt( sum_j |w_ij|^2 ||H_i|| )

    where now w_i are unit vectors * 0.5 (for the rotation gates).

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)

    Returns:
        float: Lipschitz bound of the trainable encoding.
    """
    n_layer, n_qubit, _ = weights.shape
    lip_bound = 2.0 * 1.0 * n_layer * n_qubit * 2.0 * 0.5 * np.pi
    return lip_bound

def trainable_encoding_predict(weights, biases, x):
    """
    Predict labels for the given data set x for the trainable encoding QML model.
    We use the tensor Z expectation value for the PQC and the heaviside function 
    as postprocessing to map to a label.

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, x_dim, x_dim)
        biases (array[float]): array of biases of shape (n_layer, n_qubit, x_dim)
        x (array[float]): 2-d array of input vectors

    Returns:
        predicted (array([int]): predicted labels for test data
    """
    predicted = []
    for i in range(len(x)):
        expect = trainable_encoding(weights, biases, x[i])
        label = np.sign(expect)
        predicted.append(label)
    return np.array(predicted)

def non_trainable_encoding_predict(weights, x):
    """
    Predict labels for the given data set x for the non trainable encoding QML model.
    We use the tensor Z expectation value for the PQC and the heaviside function 
    as postprocessing to map to a label.

    Args:
        weights (array[float]): array of weights of shape (n_layer, n_qubit, 3)
        x (array[float]): 2-d array of input vectors

    Returns:
        predicted (array([int]): predicted labels for test data
    """
    predicted = []
    for i in range(len(x)):
        expect = non_trainable_encoding(weights, x[i])
        label = np.sign(expect)
        predicted.append(label)
    return np.array(predicted)

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy score, i.e., correct predictions / total predictions.

    Args:
        y_true (array[float]): 1-d array of targets
        y_predicted (array[float]): 1-d array of predictions

    Returns:
        score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred
    return score.sum() / len(y_true)