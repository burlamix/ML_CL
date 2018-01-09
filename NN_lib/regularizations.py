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
            sys.exit("Regularization function undefined")

class Regularization:

    def __init__(self, f, dxf):
        self.f = f
        self.dxf = dxf

    def __str__(self):
        return str(self.f.__name__)

    def __repr__(self):
        return str(self.f.__name__)

def l2(W, rlambda=0.0):
    #L2 regularization
    return rlambda*np.sum((np.square(W)))

def l1(W, rlambda=0.0):
    #L1 regularization. Note that this leads to a more
    #sparse result than L2.
    return (np.abs(W)).sum()*rlambda

def EN(W,rlambda):
    #Elastic net regularization. Refer to
    #http://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf
    #to understand why you should use this when your computational resources allow to.
    return l1(W,rlambda[0])+l2(W,rlambda[1])

def l2reguldx(W, rlambda=0.0):
    #Derivative of L2 regularization
    return 2*(W*rlambda)

def l1reguldx(W, rlambda=0.0):
    #Derivative of L1 regularization. Note that it is not differentiable in
    #0, we therefore use a subgradient(0 in this case).
    return rlambda*(np.sign(W))

def ENdx(W, rlambda):
    #Derivative of Elastic Net regularization. Note that it is not differentiable in
    #0, we therefore use a subgradient(0 in this case).
    return l1reguldx(W, rlambda[0]) + l2reguldx(W,rlambda[1])

reguls = dict()
reguls["L2"] = Regularization(l2, l2reguldx)
reguls["L1"] = Regularization(l1, l1reguldx)
reguls["EN"] = Regularization(EN, ENdx)