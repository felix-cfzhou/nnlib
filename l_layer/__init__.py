from utils.initialize import initialize_parameters
from utils.update import update_parameters
from utils.cost import cross_entropy
from l_layer.forward import model_forward
from l_layer.backward import model_backward


class LLayer:
    """
    Simple model of arbitrary depth and complexity
    """

    def __init__(self):
        pass

    def fit_params(self, X, Y, layers_dims, num_iterations, learning_rate, verbose=True):
        self.X = X
        self.Y = Y
        self.layers_dims = layers_dims
        self.parameters = initialize_parameters(layers_dims)
        self.costs = []

        for i in range(0, num_iterations):
            AL, caches = model_forward(self.X, self.parameters)
            grads = model_backward(AL, self.Y, self.parameters, caches)
            self.parameters = update_parameters(self.parameters, grads, learning_rate)
            cost = cross_entropy(AL, self.Y)
            if i % 100 == 0:
                print(cost)
                self.costs.append(cost)

    def verify_cost(self, X_test, Y_test):
        AL = model_forward(X_test, self.parameters)
        cost = cross_entropy(AL, Y_test)

        return cost

    def predict(self, X):
        AL = model_forward(X, self.parameters)
        AL[AL >= 0.5] = 1
        AL[AL < 0.5] = 0

        return AL
