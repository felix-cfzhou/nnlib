import numpy as np


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
