import numpy as np


def calculate_gradient(W: np.array) -> np.array:
    """
    Calculate gradient of below function:
        f = 0.1*w1^2 + 2*w2^2
            df/dw1 = 0.02*w1
            df/dw2 = 4*w2
    Args:
        W (np.array):

    Returns:
        dw np.array: _description_
    """
    dw1 = W[0] * 0.2
    dw2 = W[1] * 4
    dw = np.array(dw1, dw2)

    return dw
