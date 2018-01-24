import numpy as np
import types
import sys
import sklearn as sk

from NN_lib.linesearches import dir_der,us_norm


class SimpleOptimizer:
    #SGD optimizer without momentum
    #TODO: merge with momentum optimizer
    def __str__(self):
        return str('sgd'+str(self.lr))

    def __repr__(self):
        return str('sgd'+str(self.lr))

    def __init__(self, lr=0.1,ls=None):
        self.ls = ls
        self.lr = lr

    def reset(self):
        pass

    def pprint(self):
        return "sgd,lr="+str(self.lr)

    def getLr(self):
        return self.lr

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        loss, grad = f(W)

        if self.ls!=None:
            actual_lr = self.ls(f, W, loss, grad)
        else:
            actual_lr = self.lr

        return W-actual_lr*grad


class Momentum:
    '''
    SGD optimizer with momentum
    Refer to: https://www.sciencedirect.com/science/article/pii/0041555364901375
    and 'A method of solving a convex programming problem with convergence rate O (1/k2)'
    for a more in-depth description
    TODO: merge with SimpleOptimizer
    '''
    def __str__(self):
        return str('momentum'+str(self.lr))

    def __repr__(self):
        return str('momentum'+str(self.lr))

    def __init__(self, lr=0.001, eps=0.9, nesterov=False, ls=None):
        self.lr = lr
        self.nesterov = nesterov
        self.eps = eps
        self.ls = ls
        self.reset()

    def reset(self):
        self.last_g = None

    def pprint(self):
        return "lr=" + str(self.lr)+",m="+str(self.eps)

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

        return (W+v)

class Adam:
    #Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    #A gradient based method, enriched with the first and second moment
    #information of past gradients.
    def __str__(self):
        return str('adam'+str(self.lr))

    def __repr__(self):
        return str('adam'+str(self.lr))

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
        return "adam:" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m.append((self.b1*self.m[self.t-1]+(1-self.b1)*(grad)))
        self.v.append((self.b2*self.v[self.t-1]+(1-self.b2)*(np.power((grad),2))))

        #Correction on the estimations as to avoid 0-bias due to initialization
        mcap = self.m[self.t]/(np.subtract(1,np.power(self.b1,self.t)))
        vcap = self.v[self.t]/(np.subtract(1,np.power(self.b2,self.t)))
        for i in range(len(vcap)):
            vcap[i] = np.sqrt(vcap[i])
        return np.subtract(W,self.lr*mcap/((vcap+self.eps)))


class Adamax:
    #Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    #Similar to adam but using the infinity norm
    def __str__(self):
        return str('adamax'+str(self.lr))

    def __repr__(self):
        return str('adamax'+str(self.lr))

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
        return "adamax:" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m.append(self.b1*self.m[self.t-1]+(1-self.b1)*(grad))

        k = self.b2*(self.v[-1] if self.v!=[] else np.zeros_like(grad))
        o = np.array([np.maximum(k[e],np.abs(grad[e])) for e in range(0,len(grad))])
        self.v.append(o)
        return W-(self.lr/(1-self.b1**self.t))*self.m[-1]/(np.array(self.v[-1])+self.eps)

class RMSProp:
    #RMSprop optimizer. It uses a running average of the past gradients.
    #Refer to https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    #for a more in-depth description.
    def __str__(self):
        return str('rmsprop'+str(self.lr))

    def __repr__(self):
        return str('rmsprop'+str(self.lr))

    def __init__(self, lr=0.001, delta=0.9):
        self.lr = lr
        self.delta = delta
        self.reset()

    def reset(self):
        self.R = None

    def pprint(self):
        return "rms:" + str(self.lr)

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        loss, grad = f(W)
        self.R = (1-self.delta)*(self.R if self.R!=None else 1)+(self.delta)*np.array(grad)**2
        return W-self.lr*np.array(grad)/(self.R+1e-6)**(1/2)


class ConjugateGradient:

    def __str__(self):
        return str('ConjugateGradient'+str(self.lr))

    def __repr__(self):
        return str('ConjugateGradient'+str(self.lr))

    def __init__(self, lr=0.001, eps=0.9, ls=None):
        self.lr = lr
        self.eps = eps
        self.ls = ls
        self.p = None
        self.last_g = None
        self.reset()

    def reset(self):
        self.p = None
        self.last_g = None
    def pprint(self):
        return "lr=" + str(self.lr)

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")


        loss, grad = f(W)
        #if np.random.random()<0.01:self.p=None

        if self.p == None:
            self.p = -grad#/us_norm(grad)
        else:
            beta = (-us_norm(grad)/(-us_norm(self.last_g)))**2
            #beta = dir_der(grad,grad)/dir_der(self.last_g,self.last_g)
            #print(beta)
            self.p = -grad + beta*self.p
        self.last_g = grad
        if self.ls!=None and 1==1:

            actual_lr = self.ls(f, W+self.lr*self.p,f(W+self.lr*self.p,only_fp=True), -self.p)
        else:
            actual_lr = self.lr
        return (W+actual_lr*self.p)

class Adine:


    def __str__(self):
        return str('adine'+str(self.lr))

    def __repr__(self):
        return str('adine'+str(self.lr))

    def __init__(self, lr=0.001, ms=0.9, mg=1.0001, e=1.0, ls = None):
        self.lr = lr
        self.ls = ls
        self.ms = ms
        self.mg = mg
        self.e = e
        self.reset()

    def reset(self):
        self.last_l = 0
        self.v = None
        self.t = 0
        self.avgl = 0

    def pprint(self):
        return "lr=" + str(self.lr)+",m="+str(self.eps)

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        self.t+=1
        loss = f(W,only_fp=True)
        self.last_l = self.avgl
        self.avgl = (self.avgl+loss)/2
        if self.avgl>self.e*self.last_l:
            m = self.ms
        else:
            m = self.mg
        _, grad = f(W+m*(self.v if self.v!=None else 0))
        self.v = (-self.lr * grad if (self.v == None) else (m*self.v-self.lr*grad))
        return (W+self.v)

optimizers = dict()

optimizers["SGD"] = SimpleOptimizer()
optimizers["adam"] = Adam()
optimizers["momentum"] = Momentum()
optimizers["adamax"] = Adamax()
optimizers["rmsprop"] = RMSProp()
optimizers["adine"] = Adine()
