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

    def __str__(self):
        return str(self.f.__name__)

    def __repr__(self):
        return str(self.f.__name__)

def l2(W, rlambda=0.0):
    return rlambda*np.sum((np.square(W)))

def l1(W, rlambda=0.0):
    return (np.abs(W)).sum()*rlambda

def EN(W,rlambda):
    return l1(W,rlambda[0])+l2(W,rlambda[1])

def ENdx(W, rlambda):
    return l1reguldx(W, rlambda[0]) + l2reguldx(W,rlambda[1])

def l1reguldx(W, rlambda=0.0):
    return rlambda*(np.sign(W))

def l2reguldx(W, rlambda=0.0):
    return 2*(W*rlambda)

reguls = dict()
reguls["L2"] = Regularization(l2, l2reguldx)
reguls["L1"] = Regularization(l1, l1reguldx)
reguls["EN"] = Regularization(EN, ENdx)