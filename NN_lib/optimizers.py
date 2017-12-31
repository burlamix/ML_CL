import numpy
import types
import autograd.numpy as np
import sys
from numpy import linalg as LA
from autograd import elementwise_grad as egrad



#Learn learning rate
#https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network
#Minkowski error as loss func
#line search https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
#TODO conjugate gradient http://matlab.izmiran.ru/help/toolbox/nnet/backpr59.html
class SimpleOptimizer:

    def __init__(self, lr=0.1):
        self.lr = lr

    def pprint(self):
        return "sgd,lr="+str(self.lr)

    def getLr(self):
        return self.lr

    def optimize(self, f, W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        loss, grad = f(W)
        return W-self.lr*grad

       # for i in range(epochs):
        #    o = NN.fp(x_in)
         #   loss, grad = NN.bp(o, y_true, x_in)
          #  print("Loss:"+str(loss))
            #TODO other constraints
            #update weight
            #for layer in NN.layers:
                #layer.weights = layer.weights+layer.gradients
class Momentum:
    def __init__(self, lr=0.001, eps=0.9, nesterov=False):
        self.lr = lr
        self.nesterov = nesterov
        self.eps = eps
        self.reset()

    def reset(self):
        self.last_g = None

    def pprint(self):
        return "lr=" + str(self.lr)+",mom="+self.eps

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        if self.nesterov:
            #If nesterov, "look ahead" first
            loss, grad = \
                (f(self.eps*self.last_g+W) if (self.last_g != None) else f(W))

            v = -self.lr*(grad)+self.eps*(self.last_g if (self.last_g != None) else 0)
        else:
            loss, grad = f(W)
            v = self.eps*(self.last_g if (self.last_g != None) else 0) - self.lr*(grad)
        self.last_g = v
        return W+v

class Adam:
    #Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    #A gradient based method, enriched with the first and second moment
    #information of past gradients. --TODO broader explaination (parameters descrp etc--
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()

    def reset(self):
        self.m = [0]
        self.v = [0]
        self.grad = 0
        self.t = 0

    def pprint(self):
        return "adam,lr=" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m.append(self.b1*self.m[self.t-1]+(1-self.b1)*(grad))
        self.v.append(self.b2*self.v[self.t-1]+(1-self.b2)*((grad)**2))

        #Correction on the estimations as to avoid 0-bias due to initialization
        mcap = self.m[self.t]/(1-self.b1**self.t)
        vcap = self.v[self.t]/(1-self.b2**self.t)
        for i in range(len(vcap)):
            vcap[i] = numpy.sqrt(vcap[i])
        return W-self.lr*mcap/(vcap+self.eps)


class Adamax:
    #Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    #--TODO broader explaination (parameters descrp etc--
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()

    def reset(self):
        self.m = [0]
        self.v = []
        self.grad = 0
        self.t = 0

    def pprint(self):
        return "adamax,lr=" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m.append(self.b1*self.m[self.t-1]+(1-self.b1)*(grad))

        k = self.b2*(self.v[-1] if self.v!=[] else numpy.zeros_like(grad))
        o = np.array([numpy.maximum(k[e],numpy.abs(grad[e])) for e in range(0,len(grad))])
        self.v.append(o)
        return W-(self.lr/(1-self.b1**self.t))*self.m[-1]/(np.array(self.v[-1])+self.eps)

class RMSProp:
    #TODO add documentation
    def __init__(self, lr=0.001, delta=0.9):
        self.lr = lr
        self.delta = delta
        self.R = None

    def pprint(self):
        return "RMSprop,lr=" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        loss, grad = f(W) #Compute the gradient
        self.R = (1-self.delta)*(self.R if self.R!=None else 1)+(self.delta)*np.array(grad)**2
        return W-self.lr*np.array(grad)/(self.R+1e-6)**(1/2)


optimizers = dict()


optimizers["SGD"] = SimpleOptimizer()
optimizers["adam"] = Adam()
optimizers["momentum"] = Momentum()
optimizers["adamax"] = Adamax()
