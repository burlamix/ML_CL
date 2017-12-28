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

def mse(target, prediction):
    #Mean squared error
    return ((prediction - target) ** 2).mean()

def msedx(target, prediction):
    return -(target - prediction)

def mae(target, prediction):
    #Mean absolute error
    return np.abs(target - prediction).mean()

def maedx(target, prediction):
    r = np.greater(prediction, target).astype("float32")
    r[r == 0] = -1
    return r

def mee(target,prediction):
    return np.sqrt(np.sum((target-prediction)**2,axis=1)).mean()

def meedx(target,prediction):
    #print("target",target)
    #print("prediction",prediction)
    a= np.expand_dims((np.sqrt(np.sum((target-prediction+1e-6)**2,axis=1))),axis=1)
    #a=np.tile(a,(1,prediction.shape[1]))
    return -(target-prediction)/a



losses = dict()
losses["mse"] = Loss(mse, msedx)
losses["mae"] = Loss(mae, maedx)
losses["mee"] = Loss(mee, meedx)

