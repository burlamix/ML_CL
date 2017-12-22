import numpy as np
import time

def mse(target, prediction):
    #Mean squared error
    return ((target - prediction) ** 2).mean()

def msedx(target, prediction):
    return -(target - prediction)

def mae(target, prediction):
    #Mean absolute error
    return np.abs(target - prediction).mean()

def maedx(target, prediction):
    r = np.greater(target, prediction).astype("float32")
    r[r == 0] = -1
    return r


def l2regul(W, rlambda):
    return rlambda*((W**2).sum())

def l1regul(W, rlambda):
    return W*rlambda.sum()

def l1reguldx(W, rlambda):
    return rlambda*(np.sign(W))

def l2reguldx(W, rlambda):
    return W*rlambda*2

reguls = dict()
reguls["L2"] = (l2regul, l2reguldx)
reguls["L1"] = (l1regul, l1reguldx)
losses = dict()
losses["mse"] = (mse, msedx)
losses["mae"] = (mae, maedx)

A = np.round(np.random.rand(2,2)*10)
B = np.round(np.random.rand(1,2)*10)
start = time.time()

#for i in range(0,1):
#    (mse(A,B))
#end = time.time()

#print("TIME:"+str(end-start))
