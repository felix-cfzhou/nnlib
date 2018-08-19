from numpy.testing import assert_allclose
from numpy import array

from nnlib.utils.cost import cross_entropy


def test_cross_entropy(random_state):
    Y = random_state.randn(1, 3) > 0
    cost = cross_entropy(
            array([[0.5002307,  0.49985831,  0.50023963]]),
            Y
            )

    assert_allclose(cost, 0.693058761039)
