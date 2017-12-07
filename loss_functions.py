import numpy as np
import time

def mse(target, prediction):
    #Mean squared error
    return ((target - prediction) ** 2).mean()

def msedx(target, prediction):
    return target - prediction

def mae(target, prediction):
    #Mean absolute error
    return np.abs(target - prediction).mean()

def maedx(target, prediction):
    r = np.greater(target, prediction).astype("float32")
    r[r == 0] = -1
    return r

losses = dict()
losses["mse"] = (mse, msedx)
losses["mae"] = (mae, maedx)

A = np.random.randn(500,500)
B = np.random.randn(500,500)
start = time.time()
for i in range(0,100):
    mse(A,B)
end = time.time()

print("TIME:"+str(end-start))
r = np.equal(A,A).astype("float32")
r[r==0]=-1
print(r)