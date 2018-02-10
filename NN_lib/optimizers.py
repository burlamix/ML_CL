import numpy as np
import types
import sys
#import sklearn as sk

from NN_lib.linesearches import dir_der,us_norm,us_norm2


class SimpleOptimizer:

    def __str__(self):
        return str('sgd'+str(self.lr))

    def __repr__(self):
        return str('sgd'+str(self.lr))

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
                 and self.ls.__eq__(other.ls)

    def __init__(self, lr=0.1,ls=None):
        self.ls = ls
        self.lr = lr

    def reset(self):
        pass

    def pprint(self):
        ls="None"
        if self.ls!=None: ls = self.ls.pprint()
        return "SGD{lr:"+str(self.lr)+",ls:"+ls+"}"

    def optimize(self, f, W):
        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")

        loss, grad = f(W)

        if self.ls!=None:
            actual_lr = self.ls.search(f, W, loss, -grad, us_norm2(grad,grad))
        else:
            actual_lr = self.lr

        return W-actual_lr*grad


class Momentum:
    '''
    SGD optimizer with momentum
    Refer to: https://www.sciencedirect.com/science/article/pii/0041555364901375
    and 'A method of solving a convex programming problem with convergence rate O (1/k2)'
    for a more in-depth description
    '''
    def __str__(self):
        return str('momentum'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.eps==other.eps and self.nesterov==other.nesterov and\
                 self.ls.__eq__(other.ls)

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

    def optimize(self,f,W):

        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")

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
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr and self.b1==other.b1 \
               and self.b2==other.b2 and\
                self.eps==other.eps and self.ls.__eq__(other.ls)

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
        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")

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
            actual_lr = self.ls.search(f, W-self.lr*mcap/((vcap+self.eps)),loss, grad)
        else:
            actual_lr = self.lr

        return np.subtract(W,actual_lr*mcap/((vcap+self.eps)))


class Adamax:
    #Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    #Similar to adam but using the infinity norm
    def __str__(self):
        return str('adamax'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr and self.b1==other.b1 \
               and self.b2==other.b2 and\
                self.eps==other.eps and self.ls.__eq__(other.ls)


    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()

    def reset(self):
        self.m = 0
        self.v = None
        self.grad = 0
        self.t = 0

    def pprint(self):
        return "Adamax{lr:" + str(self.lr)+",b1:"+str(self.b1)+",b2:"+str(self.b2)+"}"

    def optimize(self, f, W):
        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m = (self.b1*self.m+(1-self.b1)*(grad))

        k = self.b2*(self.v if self.v is not None else np.zeros_like(grad))
        o = np.empty_like(grad)
        for e in range(0,len(o)):
            o[e] = np.maximum(k[e],np.abs(grad[e]))
        self.v = o
        return W-(self.lr/(1-self.b1**self.t))*self.m/(np.array(self.v)+self.eps)

class RMSProp:
    #RMSprop optimizer. It uses a running average of the past gradients.
    #Refer to https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    #for a more in-depth description.
    def __str__(self):
        return str('rmsprop'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.delta==other.delta and\
                 self.ls.__eq__(other.ls)

    def __init__(self, lr=0.001, delta=0.9, ls=None):
        self.lr = lr
        self.delta = delta
        self.ls = None
        self.reset()

    def reset(self):
        self.R = None

    def pprint(self):
        return "RMSprop{lr:" + str(self.lr)+",d:"+str(self.delta)+"}"

    def optimize(self, f, W):
        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")
        loss, grad = f(W)
        self.R = (self.delta)*(self.R if not (self.R is None) else 0)+(1-self.delta)*np.array(grad)**2
        return W-self.lr*np.array(grad)/((self.R)**(1/2)+1e-6)


class ConjugateGradient:

    def __str__(self):
        return str('ConjugateGradient'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.beta_f==other.beta_f and self.restart==other.restart and\
                 self.ls.__eq__(other.ls)

    def __init__(self, lr=0.001, beta_f="FR", restart=-1,ls=None):
        self.lr = lr
        self.ls = ls
        self.p = None
        self.last_g = None
        self.t = 0
        self.beta_f = beta_f
        self.restart = restart
        self.reset()

    def reset(self):
        self.p = None
        self.last_g = None

    def pprint(self):
        ls="None"
        if self.ls!=None:
            ls = self.ls.pprint()
            lrn = ""
        else:
            lrn = "lr:"+str(self.lr)+","
        return "ConjugateGrad("+self.beta_f+"){"+lrn+"restart:"+str(self.restart)+",ls:"+ls+"}"

    def getLr(self):
        return self.lr

    def optimize(self,f,W):

        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")

        if self.restart>0:
            if (np.mod(self.t,self.restart)==0):self.p = None

        self.t+=1

        loss, grad = f(W)

        if self.p is None:
            self.p = -grad
        else:
            beta = us_norm2(grad,grad)/(us_norm2(self.last_g,self.last_g)+1e-7)
            if self.beta_f == "PR":
                beta = max(
                    0,us_norm2(grad,(grad-self.last_g))/(us_norm2(self.last_g,self.last_g)+1e-7))
            #print(beta)
            self.p = -grad + beta*self.p
        self.last_g = grad
        if self.ls!=None:
            dir_norm=us_norm2(-self.p,grad)
            actual_lr = self.ls.search(f, W,loss, self.p, dir_norm)
            #print('actual',actual_lr)
            #print('actual',actual_lr)
        else:
            actual_lr = self.lr
        return (W+actual_lr*self.p)

class Adine:


    def __str__(self):
        return str('adine'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.ms==other.ms and self.mg==other.mg and \
                    self.e == other.e and\
                        self.ls.__eq__(other.ls)

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

        #if not(isinstance(f, types.FunctionType)):
        #    sys.exit("Provided function is invalid")
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
