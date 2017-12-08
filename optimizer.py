import numpy as np
def sigmoid(x):
    #print(x)
    return 1/(1+np.exp(-x))

def sigmoddxf(x):
    return x*(1-x)

def linear(x):
    return x

def lineardxf(x):
    return 1

activations = dict()
activations["linear"] = (linear, lineardxf)


class Activation:

    def __init__(self,f,dxf,gradf):
        self.f = f
        self.dxf = dxf

class SimpleOptimizer:

    def __init__(self,lr):
        self.lr = lr
    def optimize(self, f, W):
        #Could use weights instead of NN
        loss, grad = f(W)
        return -self.lr*grad

       # for i in range(epochs):
        #    o = NN.fp(x_in)
         #   loss, grad = NN.bp(o, y_true, x_in)
          #  print("Loss:"+str(loss))
            #TODO other constraints
            #update weight
            #for layer in NN.layers:
                #layer.weights = layer.weights+layer.gradients

optimizers = dict()
optimizers["SGD"] = SimpleOptimizer(lr=0.01)
