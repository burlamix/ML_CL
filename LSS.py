
from NN_lib import optimizers
import numpy as np



def g(W, only_fp=False):
    r = 100*((W[1]-W[0]**2)**2)+(W[0]-1)**2
    if only_fp:
        return r
    else:
        return r,  np.array([2*(200*W[0]**3 -200*W[0]*W[1]+W[0]-1), 200*(W[1]-W[0]**2)])

lso = optimizers.RMSProp(lr=0.0001)
lso = optimizers.Momentum(lr=0.00001)
lso = optimizers.Adine(lr=0.00001, ms=0.9, mg=1.0001, e=1.0, ls = None)
lso = optimizers.Momentum(lr=0.00001)

initW = np.array([0,0])

for i in range(0,5000):
    newW = lso.optimize(g, initW)
   # lso.reset()
    initW = newW
    print('final',g(initW,only_fp=True))

print('final',g(initW,only_fp=True))
print('at',initW)
