import numpy as np


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation

    Arguments:
    dZ -- Gradients of cost with respect to linear output of current layer l
    cache -- Tuple of values (A_prev, W) coming from forward prop in current layer

    Returns:
    dA_prev -- Gradient of cost with respect to activation of previous layer (l-1)
    dW -- Gradient of cost with respect to W (current layer l)
    db -- Gradientof cost with respect to b (current layer l)
    """

    A_prev, W = cache
    m = A_prev.shape[1]

    dW = np.matmul(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.matmul(W.T, dZ)

    return dA_prev, dW, db


def linear_backward_activation(dA, cache, backward_func):
    """
    Implement backward propagation of entire layer

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple ((A_prev, W), A)
    backward_func -- calculates derivative of activation
    """

    linear_cache, activation_cache = cache

    dZ = backward_func(dA, linear_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
