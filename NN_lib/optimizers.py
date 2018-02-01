import numpy as np
import types
import sys
#import sklearn as sk

from NN_lib.linesearches import dir_der,us_norm,us_norm2


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
        ls="None"
        if self.ls!=None: ls = self.ls.__name__
        return "SGD{lr:"+str(self.lr)+",ls:"+ls+"}"

    def getLr(self):
        return self.lr

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        loss, grad = f(W)

        if self.ls!=None:
            actual_lr = self.ls(f, W, loss, -grad, us_norm2(grad,grad))
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
        nesterov=""
        if self.nesterov: nesterov="(nesterov)"
        return "Momentum"+nesterov+"{lr:" + str(self.lr)+",m:"+str(self.eps)+"}"

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        if self.nesterov:
            #If nesterov, "look ahead" first
            loss, grad = \
                (f(self.eps*self.last_g+W) if (not(self.last_g is None)) else f(W))
            v = -self.lr*(grad)+self.eps*(self.last_g if (not(self.last_g is None)) else 0)
        else:
            loss, grad = f(W)
            v = self.eps*(self.last_g if (not(self.last_g is None)) else 0) - self.lr*(grad)
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

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8,ls=None):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()
        self.ls = ls

    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 0
        self.ls=None

    def pprint(self):
        return "Adam{lr:" + str(self.lr)+",b1:"+str(self.b1)+",b2:"+str(self.b2)+"}"

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient
        #print('grad',us_norm(grad))
        #First and second moment estimation(biased by b1 and b2)
        self.m = ((self.b1*self.m+(1-self.b1)*(grad)))
        self.v = ((self.b2*self.v+(1-self.b2)*(np.power((grad),2))))

        #Correction on the estimations as to avoid 0-bias due to initialization
        mcap = self.m/(np.subtract(1,np.power(self.b1,self.t)))
        vcap = self.v/(np.subtract(1,np.power(self.b2,self.t)))

        for i in range(len(vcap)):
            vcap[i] = np.sqrt(vcap[i])

        if self.ls!=None:
            loss,grad = f(W-self.lr*mcap/((vcap+self.eps)))#00118553246015
            actual_lr = self.ls(f, W-self.lr*mcap/((vcap+self.eps)),loss, grad)
        else:
            actual_lr = self.lr

        return np.subtract(W,actual_lr*mcap/((vcap+self.eps)))


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
        return "Adamax{lr:" + str(self.lr)+",b1:"+str(self.b1)+",b2:"+str(self.b2)+"}"

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
        return "RMSprop{lr:" + str(self.lr)+",d:"+str(self.delta)+"}"

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        loss, grad = f(W)
        self.R = (self.delta)*(self.R if not (self.R is None) else 0)+(1-self.delta)*np.array(grad)**2
        return W-self.lr*np.array(grad)/((self.R)**(1/2)+1e-6)


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
        self.t = 0
        self.reset()

    def reset(self):
        self.p = None
        self.last_g = None

    def pprint(self):
        ls="None"
        if self.ls!=None: ls = self.ls.__name__
        return "ConjugateGrad(Fletcher-Reeves){lr:" + str(self.lr)+\
        ",eps:"+str(self.eps)+",ls:"+ls+"}"

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        self.t+=1

        loss, grad = f(W)
        if (np.mod(self.t,1)==5):self.p = None#/-us_norm(grad)

        if self.p is None:
            self.p = -grad#/(-us_norm(grad))
        else:
            #beta = -us_norm(grad)#/(-us_norm(self.last_g))
            #beta = -grad/(-us_norm(grad))
            beta = us_norm2(grad,grad)/(us_norm2(self.last_g,self.last_g)+1e-7)
            #print(beta)
            self.p = -grad + beta*self.p
        self.last_g = grad
        if self.ls!=None:
            #loss,newgrad= f(W+self.lr*self.p)
            if np.abs(us_norm(grad))>0:
                #print('val before',loss)
                dir_norm=us_norm2(-self.p,grad)
                actual_lr = self.ls(f, W,loss, self.p, dir_norm)
                #print('actual',actual_lr)
            else:
                actual_lr = self.lr
            #print('actual',actual_lr)
        else:
            actual_lr = self.lr

        f2 = f(W + actual_lr * (self.p),only_fp=True)
        #if np.random.randn() > -0.5:
        #    self.p = -grad/(-us_norm(grad))
       # if (np.abs(f2)>np.abs(loss)):
       #     self.p = -grad/(-us_norm(grad))
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
        return "Adine{lr:" + str(self.lr)+",ms:"+str(self.ms)+",mg:"+str(self.mg)+",t:"+str(self.e)+"}"

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
        _, grad = f(W+m*(self.v if not(self.v is None) else 0))
        self.v = (-self.lr * grad if (self.v is None) else (m*self.v-self.lr*grad))
        return (W+self.v)

class BFGS():
    def __str__(self):
        return str('BFGS'+str(self.lr))

    def __repr__(self):
        return str('BFGS'+str(self.lr))

    def __init__(self, lr=0.001, eps=0.9, ls=None):
        self.lr = lr
        self.V = None
        self.lastg = None
        self.reset()

    def reset(self):
        self.V = None
        self.lastg = None


    def pprint(self): #if np.random.random()<0.01:self.p=None

        return "BFGS(DFP){lr:" + str(self.lr)+"}"

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")

        loss, grad = f(W)
        if self.V == None:
            pk = -grad
        else:
            pk = dir_der(self.V, grad)
        sk = self.lr * pk
        if self.lastg == None:
            yk = grad
        else:
            yk = grad - self.lastg

        if self.V == None:
            Ak = -us_norm(yk)/-us_norm(yk)
            Bk = -us_norm(sk)/dir_der(-sk,yk)
        else:
            Ak = 1

        return (W+self.lr*pk)

optimizers = dict()

optimizers["SGD"] = SimpleOptimizer()
optimizers["adam"] = Adam()
optimizers["momentum"] = Momentum()
optimizers["adamax"] = Adamax()
optimizers["rmsprop"] = RMSProp()
optimizers["adine"] = Adine()
optimizers["BFGS"] = BFGS()
