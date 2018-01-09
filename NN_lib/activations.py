import sys
import numpy as np

class Activation:

    def __init__(self, f, dxf):
        self.f = f
        self.dxf = dxf

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

def tanhdx(x):
    return (1 - (np.power(x, 2)))


def relu(x):
    return np.maximum(x, 0)

def reludx(x):
    return 1 * (x > 0)


def softmax(x):
    #Thanks to stackoverflow for suggestions on how to
    #avoid numerical errors
    e = np.exp(x - np.max(x, axis=-1)[..., None])
    e /= np.sum(e, axis=-1)[..., None]
    #Clip to avoid numerical errors
    e[e < 1e-10] = 1e-10
    return e


def softmaxdx( a):
    #Again thanks to stackoverflow
    return [np.diag(e) for e in a[..., None, :] * (np.eye(a.shape[-1], dtype=a.dtype) -
                              a[..., None])]


activations = dict()
activations["linear"] = Activation(linear, lineardxf)
activations["sigmoid"] = Activation(sigmoid, sigmoddxf)
activations["tanh"] = Activation(tanh, tanhdx)
activations["softmax"] = Activation(softmax, softmaxdx)
activations["relu"] = Activation(relu, reludx)
