import numpy as np


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python list containing tuples of dimensions of each layer in the network

    Returns:
    parameters -- python dictionary containing lists of Weights "W[]" and list of bias vectors "b[]"
    """

    parameters = dict(W=[], b=[])
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W"][l] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters["b"][l] = np.zeros((layer_dims[l], 1))

    return parameters
