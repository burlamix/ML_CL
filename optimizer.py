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

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanhdx(x):
    return (1-(np.power(tanh(x),2)))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def softmaxdx(x):
    pass

class Activation:

    def __init__(self,f,dxf):
        self.f = f
        self.dxf = dxf

activations = dict()
activations["linear"] = Activation(linear, lineardxf)
activations["sigmoid"] = Activation(sigmoid, sigmoddxf)
activations["tanh"] = Activation(tanh, tanhdx)
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
