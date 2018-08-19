from numpy import array
from numpy.testing import assert_allclose


def test_seed(random_state):
    ex_arr = random_state.randn(1, 2)

    assert_allclose(
            ex_arr,
            array([[1.62434536, -0.61175641]])
            )
