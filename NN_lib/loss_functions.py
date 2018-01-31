import numpy as np
import sys

def validate_loss(loss_fun):
    # Check whether the user provided a properly formatted loss function
    if isinstance(loss_fun, Loss):
        return loss_fun
    else:
        # Otherwise check whether a the specified loss function exists
        try:
            return losses[loss_fun]
        except KeyError:
            sys.exit("Loss function undefined")

class Loss:

    def __init__(self,f,dxf):
        self.f = f
        self.dxf = dxf

    def __str__(self):
        return str(self.f.__name__)

    def __repr__(self):
        return str(self.f.__name__)


def mse(target, prediction):
    #Mean squared error loss function
    return ((prediction - target) ** 2).mean()

def msedx(target, prediction):
    #Derivate of mean squared error loss function
    return -(2*(target - prediction))/target.shape[1]

def mae(target, prediction):
    #Mean absolute error loss function
    return np.abs(target - prediction).mean()

def maedx(target, prediction):
    #Derivative of absolute error loss function. Note that it is not
    #differentiable in 0, we therefore use a subgradient.
    r = np.greater(prediction, target).astype("float32")
    r[r == 0] = -1
    return r/target.shape[1]

def mee(target,prediction):
    #Mean Euclidean Error loss function
    return np.sqrt(np.sum((target-prediction)**2,axis=1)).mean()

def meedx(target,prediction):
    #Derivative of Mean Euclidean Error loss function
    a= np.expand_dims((np.sqrt(np.sum((target-prediction)**2,axis=1))),axis=1)
    return -((target-prediction)/(a+1e-8))


losses = dict()
losses["mse"] = Loss(mse, msedx)
losses["mae"] = Loss(mae, maedx)
losses["mee"] = Loss(mee, meedx)
