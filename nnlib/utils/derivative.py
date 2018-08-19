def sigmoid_backward(dA, A):
    """
    partial derivative of single SIGMOID unit

    Arguments:
    dA -- post-activation gradient
    cache -- A, the post-activation matrix

    Returns:
    dZ -- gradient of cost with respect to Z
    """

    dZ = dA * A * (1-A)

    return dZ


def relu_backward(dA, A):
    """
    partial derivative of single RELU unit

    Arguments:
    dA -- post-activation gradient
    cache -- A, the post-activation matrix

    Returns:
    dZ -- gradient of cost with respect to Z
    """

    dZ = dA[A <= 0] = 0

    return dZ
