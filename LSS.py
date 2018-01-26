
from NN_lib import optimizers
import numpy as np
from NN_lib import linesearches
def matyas_fun(W, only_fp=False):
	
    r = 0.26 * (W[0][0]**2 + W[0][1]**2)  -0.48*W[0][0]*W[0][1]; 

    if only_fp:
        return r
    else:
        return r,  np.array(np.array([[ 0.52*W[0][0]-0.48*W[0][1]  ,  0.52*W[0][1]-0.48*W[0][0] ] ]))

def g(W, only_fp=False):
    r = 100*((W[1]-W[0]**2)**2)+(W[0]-1)**2

    if only_fp:
        return r
    else:
        return r,  np.array([2*(200*W[0]**3 -200*W[0]*W[1]+W[0]-1), 200*(W[1]-W[0]**2)])


amg = linesearches.armj_wolfe(m1=1e-4, m2=0.9, lr=0.00001, min_lr=1e-11, scale_r=0.000001, max_iter=1000)


lso = optimizers.RMSProp(lr=0.0001)
lso = optimizers.Momentum(lr=0.00001)
lso = optimizers.Adine(lr=0.01, ms=0.9, mg=1.0001, e=1.0, ls = None)
lso = optimizers.Momentum(lr=0.01,eps=0.6)
lso = optimizers.Adam(lr=0.1)
#lso = optimizers.SimpleOptimizer(lr=0.01,ls= amg)
lso = optimizers.RMSProp(lr=0.1)
lso = optimizers.SimpleOptimizer(lr=0.01,ls= amg)

initW = np.array([[12,34.2]])

for i in range(0,500):
    newW = lso.optimize(matyas_fun, initW)
   # lso.reset()
    initW = newW
    print('final',matyas_fun(initW,only_fp=True))

print('final',matyas_fun(initW,only_fp=True))
print('at',initW)

