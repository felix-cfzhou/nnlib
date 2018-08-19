from nnlib.utils.initialize import initialize_parameters


def test_initialize_parameters():
    layers_dims = [5, 4, 3]
    params = initialize_parameters(layers_dims)

    for i in range(len(layers_dims)-1):
        assert(params["W"][i+1].shape == (layers_dims[i+1], layers_dims[i]))
        assert(params["b"][i+1].shape == (layers_dims[i+1], 1))
