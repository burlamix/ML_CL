import sys
import numpy as np


def validate_activation(activation):
    # Check whether the user provided a properly formatted loss function
    if isinstance(activation, Activation):
        return activation
    else:
        # Otherwise check whether a the specified loss function exists
        try:
            return activations[activation]
        except KeyError:
            sys.exit("Activation undefined")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoddxf(x):
    return x * (1 - x)


def linear(x):
    return x


def lineardxf(x):
    return np.ones_like(x)


def tanh(x):
    return np.tanh(x)
    # return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def tanhdx(x):
    return (1 - (np.power(x, 2)))


def softmax1(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # print(e_x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)  # only differencen


def softmaxdx1(signal):
    J = - signal[..., None] * signal[:, None, :]  # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = signal  # diagonal
    # print(J.shape)
    return J.sum(axis=2)  # sum across-rows for each sample


def relu(x):
    return np.maximum(x, 0)


def reludx(x):
    return 1 * (x > 0)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1)[..., None])
    # note: shifting everything down by max (doesn't change
    # result, but can help avoid numerical errors)

    e /= np.sum(e, axis=-1)[..., None]

    # clip to avoid numerical errors
    e[e < 1e-10] = 1e-10
    return e


def softmaxdx( a):
    return [np.diag(e) for e in a[..., None, :] * (np.eye(a.shape[-1], dtype=a.dtype) -
                              a[..., None])]
    return np.diag(a[..., None, :] * (np.eye(a.shape[-1], dtype=a.dtype) -
                              a[..., None]),axis=1)

def loss(x):
    num_classes = x.shape[0]
    num_train = x.shape[1]
    for i in range(num_train):
      for j in range(num_classes):
        p = np.exp(f_i[j])/sum_i
        dW[j, :] += (p-(j == y[i])) * X[:, i]

class Activation:

    def __init__(self, f, dxf):
        self.f = f
        self.dxf = dxf


activations = dict()
activations["linear"] = Activation(linear, lineardxf)
activations["sigmoid"] = Activation(sigmoid, sigmoddxf)
activations["tanh"] = Activation(tanh, tanhdx)
activations["softmax"] = Activation(softmax, softmaxdx)
activations["relu"] = Activation(relu, reludx)
