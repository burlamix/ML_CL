#import numpy as np
import types
import autograd.numpy as np

from autograd import elementwise_grad as egrad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoddxf(x):
    return sigmoid(x)*(1-sigmoid(x))

def linear(x):
    return x

def lineardxf(x):
    return 1

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanhdx(x):
    return (1-(np.power(tanh(x),2)))

def softmax(x):
    e_x = np.exp( x - np.max(x,axis=1,keepdims=True))
    #print(e_x)
    return e_x / np.sum(e_x,axis=1,keepdims=True) # only differencen

def softmaxdx(signal):
    J = - signal[..., None] * signal[:, None, :] # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = signal # diagonal
    #print(J.shape)
    return J.sum(axis=1) # sum across-rows for each sample




class Activation:

    def __init__(self,f,dxf):
        self.f = f
        self.dxf = dxf

ss = egrad(softmax)
activations = dict()
activations["linear"] = Activation(linear, lineardxf)
activations["sigmoid"] = Activation(sigmoid, sigmoddxf)
activations["tanh"] = Activation(tanh, tanhdx)
activations["softmax"] = Activation(softmax, ss)
#TODO add relu
#If only linear acts do we have exact sol?
#Learn learning rate
#https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network
#Minkowski error as loss func
#line search https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
#TODO conjugate gradient http://matlab.izmiran.ru/help/toolbox/nnet/backpr59.html
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
