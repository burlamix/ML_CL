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

def armj_w(f, W, phi0, grad, m1, m2, tau, max_iter, min_lr, max_lr):
    i = 1
    a_0 = 0
    a_max = 1
    a_i = (a_max-a_0)/50
    
    #phip0- directional derivaive, norm of gradient
    phip0 = us_norm(grad)


    phi_ipast = None
    a_ipast = None

    while max_iter>0:
        #phi_i value of function where we test to go
        #gradp gradient where we test to go
        phi_i, gradp = f(W-a_i*grad)

        #test armijo strong wolfe condiction
        if phi_i>phi0+m1*a_i*phip0 or (phi_ipast!=None and phi_i>=phi_ipast):
            a_star,max_iter = zoom(a_ipast,a_i,phipg,phip0,phi0,grad,max_iter)
            return a_star

        phipg = us_norm(gradp)
        #phipp norm of the gradient where we go

        if (np.abs(phipg)<= -(m2*phip0)):
            return a_i
        elif phipg >=0:
            return zoom(a_i,a_ipast)
        else:
            a_ipast = a_i
            a_i = (a_max + a_i)/deno    

        phi_ipast = phi_i


#do quadrati interpolation
def quad_interpol(f,W,a_i,a_ipast,phipg,phip0):
    safe_g = 1e-2
    a = (a_ipast*phipg-phip0*a_i)(phipg-phip0)
    a = max(a_ipast*(1+safe_g),min(a_i*(1+safe_g),a))

    return a


def zoom(a_ipast,a_i,phipg,phip0,phi0,grad,max_iter):

    a_j = a_i
    a_j = quad_interpol(a_i,a_ipast,a,phipg,phip0)
    #
    phi_a_j, grad_a_j = f(W-a_j*grad)

    phipg_a_j = us_norm(grad_a_j)


    if phi_a_j>phi0+m1*a_j*phip0 or (phi_j>=a_ipast):
        a_i=a_j
    else:
        if (np.abs(phipg_a_j) <= -(m2*phip0)):
            return a_j
        elif (phipg_a_j*(a_ipast-a_i)>=0 ):
            a_i=a_ipast
        a_ipast=a_j





def back_track( f, W, phi0 , grad , ass , m1 , tau,max_iter , min_lr ):

    #Black magic
    print(grad.shape)
    phip0 = us_norm(grad)


    while max_iter>0 and ass > min_lr:
        phia = f(W-ass*grad,only_fp=True)
        if phia <= phi0 + m1 * ass * phip0:
            break
        
        ass = ass * tau
        max_iter -= 1

    return ass


def us_norm(x):
    x = -(np.linalg.norm(
    np.concatenate(
        [x[i].reshape(x[i].shape[1] * x[i].shape[0], 1) for i in range(0, len(x))])
        )
    )
    return x




class LineSearchOptimizer:
    def __str__(self):
        return str('ls'+str(self.lr))

    def __repr__(self):
        return str('ls'+str(self.lr))

    def __init__(self, lr=0.1, eps=1e-5, ls=None, m1=1e-7, max_iter=10, r=0.3):
        self.lr = lr
        self.eps = eps
        self.ls = ls
        self.m1 = m1
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
            actual_lr = back_track(f, W, loss, grad, self.lr,self.m1,self.r, self.max_iter, self.eps)
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
