import sys
import os
sys.path.append(os.getcwd()+"/NN_lib")
import optimizers
import numpy as np



def g(W, only_fp=False):
    r = 100*((W[1]-W[0]**2)**2)+(W[0]-1)**2
    if only_fp:
        return r
    else:
        return r,  np.array([2*(200*W[0]**3 -200*W[0]*W[1]+W[0]-1), 200*(W[1]-W[0]**2)])

lso = optimizers.LineSearchOptimizer(lr=310.33, eps=1e-8)
initW = np.array([132,32])

for i in range(0,1000):
    #print('initial',g(initW,only_fp=True))
    newW = lso.optimize(g, initW)

    initW = newW
    print('final            ',g(initW,only_fp=True))

print('final            ',g(initW,only_fp=True))
print('at',initW)
