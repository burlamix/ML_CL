import sys
sys.path.append("/home/simone/Documents/universita/Magistrale/ML/ambiente_virtuale/ML_CL/NN_lib")
import optimizers
import numpy as np



def g(W, only_fp=False):
    r = np.power(W[0,0],2) + np.power(W[0,1],2)+np.power(W[1,0],2) + np.power(W[1,1],2)
    if only_fp:
        return r
    else:
        return r,  np.array([[2*W[0,0],2*W[0,1]],[2*W[1,0],2*W[1,1]]])

lso = optimizers.LineSearchOptimizer(lr=1.33, eps=1e-8)
initW = np.array([[1,3],[4,5]])

for i in range(0,10):
    #print('initial',g(initW,only_fp=True))
    newW = lso.optimize(g, initW)

    initW = newW

print('final',g(initW,only_fp=True))
