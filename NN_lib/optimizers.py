import numpy as np
import types
import sys
import sklearn as sk

def bisection( f, W,grad,eps):
    a_minus = 0
    a_plus = 10
    max_iter = 10
    while (max_iter>0):
        max_iter-=1
        alpha = np.average([a_minus,a_plus])
        val, v = f(W-alpha*grad)
        if (np.abs(v)<=eps):
            break
        if (v>0):
            a_minus = alpha
        else:
            a_plus = alpha
    return alpha


#phi0- where we are - value of loss funcion
#grad- gradient of loss function where we are
# ass-
#m1- armj_W param
#m2 armj_w param
#max_iter num of max iteration 
#min_lr, max_lr min and max value for learing rate

#phi(a_i) = where we'd go

def armj_w(f, W, phi0, grad, m1, m2, max_iter, lr, mina, tau):
    ass = lr
    
    #phip0- directional derivaive, norm of gradient
    phip0 = us_norm(grad)
    while max_iter>0:
        #phi_i value of function where we test to go
        #gradp gradient where we test to go
        phia, phips_p = f(W-ass*grad)
        phips = us_norm2(grad,phips_p)

        #test armijo strong wolfe condiction
        if phia<=phi0+m1*ass*phip0 and (np.abs(phips)<=-m2*phip0):
            a = ass
            return a
        if phips>=0:
            break
        else:
            ass=ass/tau
            max_iter-=1

    am = 0
    a = ass
    sf=1e-3
    phipm = phip0
    while(max_iter>0) and ((ass-am)>mina and phips>1e-12):
        a = (am * phips - ass * phipm) / (phips - phipm)
        #print(a)
        temp = min([ass * (1 - sf), a])
        #print(temp)
        a = max([am * (1 + sf), temp])
        #print(a)
       # exit(1)
        #input('/.')
        phia,phipp = f(W-a*grad)
        #phip = us_norm(phipp)
        phip = us_norm2(grad,phipp)
        #print(phip)
        #print(m2*np.abs(phip0))
        if((phia<=phi0 + m1*a*phip0) and (np.abs(phip)<=-m2*(phip0))):
            return a
        if phip<0:
            am = a
            phipm = phip
        else:
            ass = a
            if ass<mina:
                return a
            phips=phip
        max_iter-=1

    print('here',max_iter)
    return a


def back_track( f, W, phi0 , grad , ass , m1 , tau,max_iter , min_lr ):

    #Black magic
    if (grad[0].shape==()):
        phip0 = -np.linalg.norm(grad)
    else:
        phip0 = us_norm(grad)


    while max_iter>0 and ass > min_lr:
        phia = f(W-ass*grad,only_fp=True)
        if phia <= phi0 + m1 * ass * phip0:
            print('ass', ass)

            break
        ass = ass * tau
        max_iter -= 1
    return ass

def us_norm2(grad, lastg):
    a = np.concatenate(
        [grad[i].reshape(grad[i].shape[1] * grad[i].shape[0], 1) for i in range(0, len(grad))])
    b = np.concatenate(
        [lastg[i].reshape(lastg[i].shape[1] * lastg[i].shape[0], 1) for i in range(0, len(lastg))])
    return np.sum(-a * b)

def us_norm(x):
    if (x[0].shape==()):
        return -np.linalg.norm(x)
    else:
        return  -(np.linalg.norm(
                np.concatenate(
                [x[i].reshape(x[i].shape[1] * x[i].shape[0], 1) for i in range(0, len(x))])
                ))



class LineSearchOptimizer:
    def __str__(self):
        return str('ls'+str(self.lr))

    def __repr__(self):
        return str('ls'+str(self.lr))

    def __init__(self, lr=0.1, eps=1e-16, ls=None, m1=0.0001, m2=0.9, max_iter=1000, r=0.9):
        self.lr = lr
        self.eps = eps
        self.ls = ls
        self.m1 = m1
        self.m2 = m2
        self.max_iter = max_iter
        self.r = r

    def reset(self):
        pass

    def pprint(self):
        return "ls,lr="+str(self.lr)

    def getLr(self):
        return self.lr

    def optimize(self, f, W):
        if not(isinstance(f, types.FunctionType)):
            sys.exit("Provided function is invalid")
        loss, grad = f(W)
        if self.ls == 'back_tracking':
            actual_lr = back_track\
                (f = f, W=W, phi0=loss, grad=grad, ass=self.lr,m1=self.m1,
                 tau=self.r, max_iter=self.max_iter, min_lr=self.eps)
        if self.ls == 'armj-wolfe':
            actual_lr = armj_w \
                (f=f, W=W, phi0=loss, grad=grad, m1=self.m1, m2=self.m2,
                     max_iter=self.max_iter, mina=0, lr = self.lr, tau=self.r)
        else:
            actual_lr=self.lr

        return W-actual_lr*grad


class SimpleOptimizer:
    #SGD optimizer without momentum
    #TODO: merge with momentum optimizer
    def __str__(self):
        return str('sgd'+str(self.lr))

    def __repr__(self):
        return str('sgd'+str(self.lr))

    def __init__(self, lr=0.1,ls=False):
        self.lr = lr
        self.ls = ls

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
        if self.ls :
            max_iter=20
            a=self.lr
            while(f(W-a,only_fp=True)>f(W-a*1.5,only_fp=True)):
                max_iter-=1
                a=a*1.5

            return (W-(a*grad))

        return W-self.lr*grad


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

    def __init__(self, lr=0.001, eps=0.9, nesterov=False):
        self.lr = lr
        self.nesterov = nesterov
        self.eps = eps
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


optimizers = dict()

optimizers["SGD"] = SimpleOptimizer()
optimizers["adam"] = Adam()
optimizers["momentum"] = Momentum()
optimizers["adamax"] = Adamax()
optimizers["rmsprop"] = RMSProp()
