import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose

from nnlib.l_layer.forward import model_forward


def test_model_forward():
    rand = RandomState(6)
    X = rand.randn(5, 4)
    W1 = rand.randn(4, 5)
    b1 = rand.randn(4, 1)
    W2 = rand.randn(3, 4)
    b2 = rand.randn(3, 1)
    W3 = rand.randn(1, 3)
    b3 = rand.randn(1, 1)

    parameters = {
            "W": {1: W1, 2: W2, 3: W3},
            "b": {1: b1, 2: b2, 3: b3},
            }

    AL, caches = model_forward(X, parameters, keep_prob=1)
    ans = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])

    assert_allclose(AL, ans)
    assert(len(caches["A"]) == 4)


def test_model_forward_dropout():
    rand = RandomState(1)
    X = rand.randn(3, 5)
    W1 = rand.randn(2, 3)
    b1 = rand.randn(2, 1)
    W2 = rand.randn(3, 2)
    b2 = rand.randn(3, 1)
    W3 = rand.randn(1, 3)
    b3 = rand.randn(1, 1)

    parameters = {
        'W': {1: W1, 2: W2, 3: W3},
        'b': {1: b1, 2: b2, 3: b3}
    }

    np.random.seed(1)
    AL, caches = model_forward(X, parameters, keep_prob=0.7)
    ans = np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]])

    assert_allclose(AL, ans, rtol=1e-05)
    assert(len(caches["A"]) == len(caches["D"])+1)
