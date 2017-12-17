import numpy
import types
import autograd.numpy as np
import sys

from autograd import elementwise_grad as egrad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoddxf(x):
    return x*(1-x)

def linear(x):
    return x

def lineardxf(x):
    return 1

def tanh(x):
    return numpy.tanh(x)
    #return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanhdx(x):
    return (1-(np.power(x,2)))

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

    def __init__(self,lr=0.1):
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
class Momentum:
    def __init__(self, mu=0.001, eps=0.9, nesterov=False):
        self.mu = mu
        self.nesterov = nesterov
        self.eps = eps
        self.reset()

    def reset(self):
        self.last_g = [0]


    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")


        #W=np.array(list(reversed(W)))            #the grad are reversed, so use a reversed W if you want do W+gradient

        if self.nesterov:

            print("-g-",self.last_g)
            print("-W-",W)
            print("-x-",self.last_g+W)

            #c'è ancora qualche problema con il nesterov
            loss, grad = f(self.last_g+W)
            #loss, grad = f(W)

            back = self.mu*self.last_g
            now = self.eps*grad
        else:
            
            loss, grad = f(W)
            
            back = self.mu*self.last_g
            now = self.eps*grad

        self.last_g = back-now
        return self.last_g
#TODO better if we return the updated weight rather than the stuff for update the weight?

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

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m.append(self.b1*self.m[self.t-1]+(1-self.b1)*grad)
        self.v.append(self.b2*self.v[self.t-1]+(1-self.b2)*(grad**2))

        #Correction on the estimations as to avoid 0-bias due to initialization
        mcap = self.m[self.t]/(1-self.b1**self.t)
        vcap = self.v[self.t]/(1-self.b2**self.t)
        for i in range(len(vcap)):
            vcap[i] = numpy.sqrt(vcap[i])
        return -self.lr*mcap/(vcap+self.eps)


optimizers = dict()


optimizers["SGD"] = SimpleOptimizer()
optimizers["adam"] = Adam()
optimizers["momentum"] = Momentum()
