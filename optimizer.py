import numpy as np
import types

def sigmoid(x):
    #print(x)
    return 1/(1+np.exp(-x))

def sigmoddxf(x):
    return x*(1-x)

def linear(x):
    return x

def lineardxf(x):
    return 1

#TODO test this
def hyperbolic_tangent(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def hyperbolic_tangentdx(x):
    return (1-(np.power(hyperbolic_tangent(x),2)))

class Activation:

    def __init__(self,f,dxf):
        self.f = f
        self.dxf = dxf

activations = dict()
activations["linear"] = Activation(linear, lineardxf)
activations["sigmoid"] = Activation(sigmoid, sigmoddxf)
activations["hyperbolic_tangent"] = Activation(hyperbolic_tangent, hyperbolic_tangentdx)


class SimpleOptimizer:

    def __init__(self,lr):
        self.lr = lr

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
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
