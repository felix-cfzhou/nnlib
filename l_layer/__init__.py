from utils.activation import relu, sigmoid
from l_layer.forward import linear_forward_activation


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
