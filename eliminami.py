
import numpy as np
def softmax(x):
    e_x = np.exp( x - np.max(x,axis=1,keepdims=True))
    #print(np.sum(e_x,axis=1,keepdims=True))
    return e_x / np.sum(e_x,axis=1,keepdims=True) # only differencen

a=np.array([[1,2,3],[11,22,33]])
print("------------")

#print(a.shape)
print("------------")
print(softmax(a))
print("------------")

print(np.sum(softmax(a)[0]))