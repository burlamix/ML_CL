from NN_lib import optimizers
import numpy as np

def g(W, only_fp=False):
    r = W**2 + 5
    if only_fp:
        return r
    else:
        return r, 2*W+3

lso = optimizers.LineSearchOptimizer(lr=0.1, eps=1e-8)
initW = 2

for i in range(0,1):
    #print('initial',g(initW,only_fp=True))
    newW = lso.optimize(g, initW)

    initW = newW

print('final',g(initW,only_fp=True))
