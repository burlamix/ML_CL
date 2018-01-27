import numpy as np
from NN_lib import optimizers

def matyas_fun(W, only_fp=False):
    r = 0.26 * (W[0][0] ** 2 + W[0][1] ** 2) - 0.48 * W[0][0] * W[0][1];

    if only_fp:
        return r
    else:
        return r, np.array(np.array([[0.52 * W[0][0] - 0.48 * W[0][1], 0.52 * W[0][1] - 0.48 * W[0][0]]]))


def rosenbrock(W, only_fp=False):
    r = 100 * ((W[0][1] - W[0][0] ** 2) ** 2) + (W[0][0] - 1) ** 2

    if only_fp:
        return r
    else:
        return r, np.array([[2 * (200 * W[0][0] ** 3 - 200 * W[0][0] * W[0][1] + W[0][0] - 1),
                            200 * (W[0][1] - W[0][0] ** 2)]])

