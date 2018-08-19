from numpy import array
from numpy.random import RandomState
from numpy.testing import assert_allclose

from nnlib.l_layer.backward import linear_backward, linear_backward_activation, model_backward
from nnlib.utils.derivative import sigmoid_backward, relu_backward
from nnlib.utils.activation import sigmoid, relu


def test_linear_backward():
    rand = RandomState(1)
    dZ = rand.randn(1, 2)
    A = rand.randn(3, 2)
    W = rand.randn(1, 3)

    dA_prev, dW, db = linear_backward(dZ, (A, W))

    assert_allclose(dA_prev, [
        [0.51822968, -0.19517421],
        [-0.40506361, 0.15255393],
        [2.37496825, -0.89445391]])
    assert_allclose(dW, [[-0.10076895, 1.40685096, 1.64992505]])
    assert_allclose(db, [[0.50629448]])


def test_linear_backward_activation_sigmoid():
    rand = RandomState(2)
    dA = rand.randn(1, 2)
    A = rand.randn(3, 2)
    W = rand.randn(1, 3)
    b = rand.randn(1, 1)  # noqa: F841
    Z = rand.randn(1, 2)
    dA_prev, dW, db = linear_backward_activation(dA, ((A, W), (Z, sigmoid(Z))), sigmoid_backward)
    assert_allclose(dA_prev, array([
        [0.11017994, 0.01105339],
        [0.09466817, 0.00949723],
        [-0.05743092, -0.00576154]]), rtol=1e-05)
    assert_allclose(dW, array([[0.10266786, 0.09778551, -0.01968084]]), rtol=1e-05)
    assert_allclose(db, array([[-0.05729622]]), rtol=1e-05)


def test_linear_backward_activation_relu():
    rand = RandomState(2)
    dA = rand.randn(1, 2)
    A = rand.randn(3, 2)
    W = rand.randn(1, 3)
    b = rand.randn(1, 1)  # noqa: F841
    Z = rand.randn(1, 2)
    dA_prev, dW, db = linear_backward_activation(dA, ((A, W), (Z, relu(Z))), relu_backward)
    assert_allclose(dA_prev, array([
        [0.44090989, 0.],
        [0.37883606, 0.],
        [-0.2298228, 0.]]), rtol=1e-05)
    assert_allclose(dW, array([[0.44513824, 0.37371418, -0.10478989]]), rtol=1e-05)
    assert_allclose(db, array([[-0.20837892]]), rtol=1e-05)


def test_model_backward():
    rand = RandomState(3)
    AL = rand.randn(1, 2)
    Y = array([[1, 0]])

    X = rand.randn(4, 2)
    W1 = rand.randn(3, 4)
    b1 = rand.randn(3, 1)
    Z1 = rand.randn(3, 2)

    A1 = rand.randn(3, 2)
    W2 = rand.randn(1, 3)
    b2 = rand.randn(1, 1)
    Z2 = rand.randn(1, 2)

    parameters = dict(
            W={1: W1, 2: W2},
            b={1: b1, 2: b2}
            )
    caches = dict(
            Z={1: Z1, 2: Z2},
            A={0: X, 1: A1, 2: sigmoid(Z2)}
            )

    grads = model_backward(AL, Y, parameters, caches)

    assert_allclose(
            grads["dW"][1],
            array([
                [0.41010002, 0.07807203, 0.13798444, 0.10502167],
                [0., 0., 0., 0.],
                [0.05283652, 0.01005865, 0.01777766, 0.0135308]]),
            rtol=1e-05
            )
    assert_allclose(
            grads["db"][1],
            array([
                [-0.22007063],
                [0.],
                [-0.02835349]])
            )
    assert_allclose(
            grads["dA"][1],
            array([
                [0.12913162, -0.44014127],
                [-0.14175655, 0.48317296],
                [0.01663708, -0.05670698]]),
            rtol=1e-05
            )
