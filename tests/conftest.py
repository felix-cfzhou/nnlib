import pytest
from numpy.random import RandomState


@pytest.fixture
def random_state():
    return RandomState(1)
