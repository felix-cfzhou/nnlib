import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose

from nnlib.utils.cost import cross_entropy


def test_cross_entropy():
    random_state = RandomState(1)
    Y = random_state.randn(1, 3) > 0
    W1 = random_state.randn(2, 3)
    b1 = random_state.randn(2, 1)
    W2 = random_state.randn(3, 2)
    b2 = random_state.randn(3, 1)
    W3 = random_state.randn(1, 3)
    b3 = random_state.randn(1, 1)
    parameters = {
            'W': {1: W1, 2: W2, 3: W3},
            'b': {1: b1, 2: b2, 3: b3}
            }
    cost = cross_entropy(
            np.array([[0.5002307,  0.49985831,  0.50023963]]),
            Y,
            parameters,
            alpha=0
            )
    assert_allclose(cost, 0.693058761039)


def test_cross_entropy_l2_regularization():
    random_state = RandomState(1)
    Y = np.array([[1, 1, 0, 1, 0]])
    W1 = random_state.randn(2, 3)
    b1 = random_state.randn(2, 1)
    W2 = random_state.randn(3, 2)
    b2 = random_state.randn(3, 1)
    W3 = random_state.randn(1, 3)
    b3 = random_state.randn(1, 1)
    parameters = {
            'W': {1: W1, 2: W2, 3: W3},
            'b': {1: b1, 2: b2, 3: b3}
            }
    cost = cross_entropy(
            np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]]),
            Y,
            parameters,
            alpha=0.1
            )

    assert_allclose(cost, 1.78648594516)
