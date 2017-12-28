import numpy as np
from sklearn.metrics import mean_squared_error
def mee(target,prediction):
    return np.sqrt(np.sum((target-prediction)**2,axis=1)).mean()

def meedx(target,prediction):
    print("target",target)
    print("prediction",prediction)
    a= np.expand_dims((np.sqrt(np.sum((target-prediction+1e-6)**2,axis=1))),axis=1)
    a=np.tile(a,(1,prediction.shape[1]))

    return (target-prediction)/a


def mse(target, prediction):
    #Mean squared error
    #np.sum(((prediction - target) ** 2)/prediction.shape[1],axis=1)
    return np.sum(np.square(prediction-target),axis=1).mean()

def msedx(target, prediction):
    return (prediction-target)


a = np.array([[4,3],[2,2],[1,1],[1,1]])#[2,1]])
b = np.array([[3,1],[1,1],[1,1],[1,1]])#,[2,1]])
xx= np.array([2,2])


print("dx mee",meedx(a,b))
#print("dx mSe",msedx(a,b))

def mee(target,prediction):
    return np.sqrt(np.sum((target-prediction)**2,axis=1)).mean()

def meedx(target,prediction):
    #print("target",target)
    #print("prediction",prediction)
    a= np.expand_dims((np.sqrt(np.sum((target-prediction+1e-6)**2,axis=1))),axis=1)
    a=np.tile(a,(1,prediction.shape[1]))

    return -(target-prediction)/a
