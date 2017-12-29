import numpy as np
import sys

def validate_regularizer(regularizer):
    if isinstance(regularizer, Regularization):
        return regularizer
    else:
        # Otherwise check whether the specified regularizer function exists
        try:
            return reguls[regularizer]
        except KeyError:
            sys.exit("regularizer function undefined")

class Regularization:
    def __init__(self, f, dxf):
        self.f = f
        self.dxf = dxf

def l2regul(W, rlambda):
    return (rlambda*(W**2)).sum()

def l1regul(W, rlambda):
    return (W*rlambda).sum()

def l1reguldx(W, rlambda):
    return rlambda*(np.sign(W))

def l2reguldx(W, rlambda):
    return W*rlambda*2

reguls = dict()
reguls["L2"] = Regularization(l2regul, l2reguldx)
reguls["L1"] = Regularization(l1regul, l1reguldx)