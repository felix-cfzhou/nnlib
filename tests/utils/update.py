import numpy as np
from numpy.testing import assert_allclose

from nnlib.utils.update import update_parameters


def test_update_parameters():
    W1 = np.ones((3, 4))
    b1 = np.ones((3, 1))
    W2 = np.ones((1, 3))
    b2 = np.ones((1, 1))
    parameters = {
            "W": {1: W1, 2: W2},
            "b": {1: b1, 2: b2}
                  }
    grads = {
            "dW": {1: W1*0.05, 2: W2*0.05},
            "db": {1: b1*0.05, 2: b2*0.05}
                  }
    new_params = update_parameters(parameters, grads, learning_rate=1)

    assert_allclose(new_params["W"][1], W1*0.95)
    assert_allclose(new_params["W"][2], W2*0.95)
    assert_allclose(new_params["b"][1], b1*0.95)
    assert_allclose(new_params["b"][2], b2*0.95)
