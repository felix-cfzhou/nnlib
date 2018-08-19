import numpy as np

from nnlib.utils.activation import relu, sigmoid


def linear_forward(A_prev, W, b):
    """
    Implement linear part of forward propagation

    Arguments:
    A_prev -- activations from previous layer or input data
    W -- weight matrix
    b -- bias vector

    Returns:
    Z -- input for activation function
    """

    Z = np.matmul(W, A_prev) + b

    return Z


def linear_forward_activation(A_prev, W, b, activation_func):
    """
    Implement forward propagation
    Arguments:
    A_prev -- activation from previous layer or input data
    W -- weight matrix
    b -- bias vector
    activation_func -- activation function (from utils)

    Returns:
    A -- output of activation function
    """

    Z = linear_forward(A_prev, W, b)
    A = activation_func(Z)

    return A


def model_forward(X, parameters):
    """
    Implement forward propagation sequence

    Arguments:
    X -- input data
    parameters -- initial weights

    Returns:
    AL -- last post-activation value
    caches -- dictionary of lists containing values computed in the forward pass
    """
    caches = dict(A=[])
    A = X
    L = len(parameters["W"])

    for l in range(1, L):
        A_prev = A
        Wl = parameters["W"][l]
        bl = parameters["b"][l]
        A = linear_forward_activation(A_prev, Wl, bl, relu)
        caches["A"][l] = A

    AL = linear_forward_activation(A, parameters["W"][L], parameters["b"][L], sigmoid)
    caches["A"][L] = AL

    return AL, caches
