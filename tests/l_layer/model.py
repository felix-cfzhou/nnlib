import numpy as np
from pytest import approx, mark

from nnlib.l_layer import LLayer


@mark.timeout(1800)
def test_llayer(cat_dataset):
    np.random.seed(1)

    train_x_orig, train_y, test_x_orig, test_y, classes = cat_dataset

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    model = LLayer()

    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1

    model.fit_params(
            train_x,
            train_y,
            layers_dims=(n_x, n_h, n_y),
            num_iterations=2500,
            verbose=False
            )

    train_acc = model.verify_accuracy(train_x, train_y)
    predictions_acc = model.verify_accuracy(test_x, test_y)

    assert(train_acc == approx(1.0))
    assert(predictions_acc == approx(0.74))
