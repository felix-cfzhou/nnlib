from numpy import array
from numpy.random import RandomState
from numpy.testing import assert_allclose

from nnlib.l_layer.backward import linear_backward, linear_backward_activation, model_backward
from nnlib.l_layer.forward import model_forward
from nnlib.utils.derivative import sigmoid_backward, relu_backward
from nnlib.utils.activation import sigmoid, relu


def test_linear_backward():
    rand = RandomState(1)
    dZ = rand.randn(1, 2)
    A = rand.randn(3, 2)
    W = rand.randn(1, 3)

    dA_prev, dW, db = linear_backward(dZ, (A, 1, W), alpha=0, keep_prob=1)

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
    dA_prev, dW, db = linear_backward_activation(dA, ((A, 1, W), (Z, sigmoid(Z))), sigmoid_backward, alpha=0, keep_prob=1)
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
    dA_prev, dW, db = linear_backward_activation(dA, ((A, 1, W), (Z, relu(Z))), relu_backward, alpha=0, keep_prob=1)
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
            A={0: X, 1: A1, 2: sigmoid(Z2)},
            D={0: 1, 1: 1}
            )

    grads = model_backward(AL, Y, parameters, caches, alpha=0, keep_prob=1)

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


def test_model_backward_l2_regularization():
    random_state = RandomState(1)
    X = random_state.randn(3, 5)
    Y = array([[1, 1, 0, 1, 0]])
    cache = (array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                    [-1.98043538,  4.1600994,  0.79051021,  1.46493512, -0.45506242]]),
             array([[0.,  3.32524635,  2.13994541,  2.60700654,  0.],
                    [0.,  4.1600994,  0.79051021,  1.46493512,  0.]]),
             array([[-1.09989127, -0.17242821, -0.87785842],
                    [0.04221375,  0.58281521, -1.10061918]]),
             array([[1.14472371],
                    [0.90159072]]),
             array([[0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                    [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
                    [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
             array([[0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                    [0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.]]),
             array([[0.50249434,  0.90085595],
                    [-0.68372786, -0.12289023],
                    [-0.93576943, -0.26788808]]),
             array([[0.53035547],
                    [-0.69166075],
                    [-0.39675353]]),
             array(
                 [[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104]]),
             array(
        [[0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]]),
        array([[-0.6871727, -0.84520564, -0.67124613]]),
        array([[-0.0126646]]))

    Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3 = cache

    parameters = dict(
            W={1: W1, 2: W2, 3: W3},
            b={1: b1, 2: b2, 3: b3}
            )
    caches = dict(
            Z={1: Z1, 2: Z2, 3: Z3},
            A={0: X, 1: A1, 2: A2, 3: sigmoid(A3)},
            D={0: 1, 1: 1, 2: 1}
            )

    AL, caches = model_forward(X, parameters, keep_prob=1)
    grads = model_backward(AL, Y, parameters, caches, alpha=0.7, keep_prob=1)

    dW1 = array([[-0.25604646,  0.12298827, - 0.28297129],
                 [-0.17706303,  0.34536094, - 0.4410571]])
    dW2 = array([[0.79276486,  0.85133918],
                 [-0.0957219, - 0.01720463],
                 [-0.13100772, - 0.03750433]])
    dW3 = array([[-1.77691347, - 0.11832879, - 0.09397446]])

    assert_allclose(grads['dW'][1], dW1)
    assert_allclose(grads['dW'][2], dW2, rtol=1e-05)
    assert_allclose(grads['dW'][3], dW3)
