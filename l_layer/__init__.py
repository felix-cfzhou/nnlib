import numpy as np

from utils.activation import relu, sigmoid
from utils.derivative import relu_backward, sigmoid_backward
from l_layer.forward import linear_forward_activation
from l_layer.backward import linear_backward_activation


class LLayer:
    """
    Simple model of arbitrary depth and complexity
    """

    @staticmethod
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

    @staticmethod
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
