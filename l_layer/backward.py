import numpy as np

from utils.derivative import relu_backward, sigmoid_backward


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


def model_backward(AL, Y, parameters, caches):
    """
    Implement backward propgation of arbitrary model

    Arguments:
    Al -- output of forward prop
    Y -- labels for data
    parameters -- dictionary of weights W, b
    caches -- dictionary of post activation values A

    Returns:
    grads -- dictionary of lists for gradients of each layer
    """

    grads = dict(dA=[], dW=[], db=[])
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    grads["dA"][L] = dAL
    dA_prev, dWL, dbL = linear_backward_activation(
            dAL,
            ((caches["A"][L-1], parameters["W"][L]), caches["A"][L]),
            sigmoid_backward
            )
    grads["dA"][L-1] = dA_prev
    grads["dW"][L] = dWL
    grads["db"][L] = dbL

    for l in reversed(range(1, L)):
        dA_prev, dWl, dbl = linear_backward_activation(
                grads["dA"][l],
                ((caches["A"][l-1], parameters["W"][l]), caches["A"][l]),
                relu_backward
                )
        grads["dA"][l-1] = dA_prev
        grads["dW"][l] = dWl
        grads["db"][l] = dbl

    return grads
