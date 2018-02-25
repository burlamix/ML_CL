import numpy as np
from NN_lib.linesearches import dir_der


class SGD:
    '''
    Basic gradient descent without momentum or any other fancy addition.
    Only a line searcher may be handed.
    '''
    def __str__(self):
        return str('sgd'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
                 and self.ls.__eq__(other.ls)

    def pprint(self):
        '''Pretty printing'''
        ls = "None"
        if self.ls!=None:
            ls = self.ls.pprint()
            lrn = ""
        else:
            lrn = "lr:"+str(self.lr)+","
        return "SGD{"+lrn+"ls:"+ls+"}"

    def __init__(self, lr=0.1,ls=None):
        '''
        :param lr: Determines the step size to use when moving in the gradient' direction.
        Note that if a line search is specified then this parameter is simply ignored.
        :param ls: Line search object. Refer to linesearches.py
        '''
        self.ls = ls
        self.lr = lr

    def reset(self):
        #Resets the state of the optimizer. SGD does not even have one though.
        pass


    def optimize(self, f, W):
        loss, grad = f(W)
        if self.ls!=None:
            actual_lr = self.ls.search(f, W, loss, -grad, dir_der(grad,grad))
        else:
            actual_lr = self.lr
        return W-actual_lr*grad


class Momentum:
    '''
    SGD optimizer with momentum.
    Refer to https://www.sciencedirect.com/science/article/pii/0041555364901375.
    or 'A method of solving a convex programming problem with convergence rate O (1/k2)'
    for a more in-depth description.
    '''
    def __str__(self):
        return str('momentum'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.eps==other.eps and self.nesterov==other.nesterov

    def pprint(self):
        '''Pretty printing'''
        nesterov=""
        if self.nesterov: nesterov="(nesterov)"
        return "Momentum"+nesterov+"{lr:" + str(self.lr)+",m:"+str(self.eps)+"}"


    def __init__(self, lr=0.001, eps=0.9, nesterov=False):
        '''
        :param lr: Determines the step size to move by.
        :param eps: Momentum parameter, the higher it is, the higher the weight given to
        past directions
        :param nesterov: Set to True for nesterov's version. That is effectively looking
        ahead first, by changing the point in which the gradient is computed.
        '''
        self.lr = lr
        self.eps = eps
        self.nesterov = nesterov
        self.reset()

    def reset(self):
        #Reset for SGD with momentum => forget the past directions
        self.last_g = None

    def optimize(self,f,W):

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

class Adine:
    '''
    An adaptive momentum gradient descent method. Characterized by two momentum terms to
    switch between adaptively.
    Refer to https://arxiv.org/pdf/1712.07424.pdf for an in-depth description.
    '''
    def __str__(self):
        return str('adine'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.ms==other.ms and self.mg==other.mg and \
                    self.e == other.e

    def pprint(self):
        '''Pretty printing'''
        return "Adine{lr:" + str(self.lr)+",ms:"+str(self.ms)+",mg:"+str(self.mg)+",t:"+str(self.e)+"}"

    def __init__(self, lr=0.001, ms=0.9, mg=1.0001, e=1.0):
        '''
        :param lr: The step size to move by.
        :param ms: The standard momentum term.
        :param mg: The greater momentum term.
        :param e: Tolerance parameter determining when to switch between the
        standard and greater momentum terms.
        '''
        self.lr = lr
        self.ms = ms
        self.mg = mg
        self.e = e
        self.reset()

    def reset(self):
        #Reset for adine => forget the weighted-sum loss and past direction.
        self.last_l = 0
        self.v = None
        self.t = 0
        self.avgl = 0

    def optimize(self,f,W):

        self.t+=1
        loss = f(W,only_fp=True) #Compute the current value of the function
        self.last_l = self.avgl
        self.avgl = (self.avgl+loss)/2 #Keep a weighted avg of the function' value progression

        if self.avgl>self.e*self.last_l: #If we are going too far in what looks like a bad
            #direction then slow down
            m = self.ms
        else:
            m = self.mg
        _, grad = f(W+m*(self.v if not(self.v is None) else 0))
        self.v = (-self.lr * grad if (self.v is None) else (m*self.v-self.lr*grad))

        return (W+self.v)

class Adam:
    '''
    Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    A gradient based method, enriched with the first and second moment
    information of past gradients.
    '''

    def __str__(self):
        return str('adam'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr and self.b1==other.b1 \
               and self.b2==other.b2 and\
                self.eps==other.eps

    def pprint(self):
        '''Pretty printing'''
        return "Adam{lr:" + str(self.lr)+",b1:"+str(self.b1)+",b2:"+str(self.b2)+"}"

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        '''
        :param lr: The step size to move by.
        :param b1: Decay rate of the moving average of the gradient.
        :param b2: Decay rate of the moving average of the squared gradient.
        :param eps: Gotta avoid those numerical issues.
        '''
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()

    def reset(self):
        #Reset for adam => forget first moment and second moment moving averages.
        self.m = 0
        self.v = 0
        self.lastf = 1000000
        self.t = 0

    def optimize(self, f, W):

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient
        #First and second moment estimation(biased by b1 and b2)
        self.m = ((self.b1*self.m+(1-self.b1)*(grad)))
        self.v = ((self.b2*self.v+(1-self.b2)*(np.power((grad),2))))

        '''
        if self.lastf < loss:
            self.m=self.m/1.5
            self.v = self.v/1.5

        else:
            self.v = self.v*1.5
            self.m = self.m*1.5
        '''

        #Correction on the estimations as to avoid 0-bias due to initialization
        mcap = self.m/(np.subtract(1,np.power(self.b1,self.t)))
        vcap = self.v/(np.subtract(1,np.power(self.b2,self.t)))

        for i in range(len(vcap)):
            vcap[i] = np.sqrt(vcap[i])


        self.lastf = loss
        return np.subtract(W,self.lr*mcap/((vcap+self.eps)))


class Adamax:
    '''
    Implementation based on https://arxiv.org/pdf/1412.6980.pdf
    Similar to adam but using the infinity norm.
    '''
    def __str__(self):
        return str('adamax'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr and self.b1==other.b1 \
               and self.b2==other.b2 and\
                self.eps==other.eps

    def pprint(self):
        '''Pretty printing'''
        return "Adamax{lr:" + str(self.lr)+",b1:"+str(self.b1)+",b2:"+str(self.b2)+"}"

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        '''
        :param lr: The step size to move by.
        :param b1: Decay rate of the moving average of the gradient.
        :param b2: Decay rate of the moving average of the squared gradient.
        :param eps: Gotta avoid those numerical issues.
        '''
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.reset()

    def reset(self):
        #Reset for adamax=> forget first moment and second moment moving averages.
        self.m = 0
        self.v = None
        self.t = 0
        self.lastf = 10e9

    def optimize(self, f, W):

        self.t+=1 #Update timestamp
        loss, grad = f(W) #Compute the gradient

        #First and second moment estimation(biased by b1 and b2)
        self.m = (self.b1*self.m+(1-self.b1)*(grad))

        k = self.b2*(self.v if self.v is not None else np.zeros_like(grad))
        o = np.empty_like(grad)

        #Infinity norm here
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
               and self.delta==other.delta

    def pprint(self):
        '''Pretty printing'''
        return "RMSprop{lr:" + str(self.lr)+",d:"+str(self.delta)+"}"

    def __init__(self, lr=0.001, delta=0.9):
        '''
        :param lr: The step size to move by.
        :param delta: Parameter controlling the weight of the past gradients running average
        '''
        self.lr = lr
        self.delta = delta
        self.reset()

    def reset(self):
        #Reset for RMSprop => forget the past gradients
        self.R = None

    def optimize(self, f, W):
        loss, grad = f(W)
        self.R = (self.delta)*(self.R if not (self.R is None) else 0)+(1-self.delta)*np.array(grad)**2
        return W-self.lr*np.array(grad)/((self.R)**(1/2)+1e-6)


class ConjugateGradient:
    '''
    Implementation of non-linear conjugate gradient methods.
    Fletcher-Reeves and Polak-Ribière variants are implemented as of now.
    '''
    def __str__(self):
        return str('ConjugateGradient'+str(self.lr))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.beta_f==other.beta_f and self.restart==other.restart and\
                 self.ls.__eq__(other.ls)

    def pprint(self):
        '''Pretty printing'''
        ls="None"
        if self.ls!=None:
            ls = self.ls.pprint()
            lrn = ""
        else:
            lrn = "lr:"+str(self.lr)+","
        return "ConjugateGrad("+self.beta_f+"){"+lrn+"restart:"+str(self.restart)+",ls:"+ls+"}"


    def __init__(self, lr=0.001, beta_f="FR", restart=-1,ls=None):
        '''
        :param lr: The step size of move by. Note that if a line search
        is specified then this parameter is simply ignored.
        :param beta_f: Specifies the version of conjugate gradient to be used.
        Possible values are "FR" for Fletcher-Reeves and "PR" for
        Polak-Ribière.
        :param restart: Specifies the number of iterations to restart after. A value
        of -1 indicates no restarts.
        :param ls: Line search object.
        '''
        self.lr = lr
        self.beta_f = beta_f
        self.restart = restart
        self.ls = ls
        self.p = None #Initial conjugate direction
        self.last_g = None #Gradient of last iteration
        self.reset()

    def reset(self):
        #Reset for Conjugate gradient => forget the past gradient and direction.
        self.p = None
        self.last_g = None
        self.t = 0

    def optimize(self,f,W):

        #If it's time to restart then forget current the direction.
        if self.restart>0:
            if (np.mod(self.t,self.restart)==0):self.p = None

        self.t+=1

        loss, grad = f(W)

        if self.p is None:
            #First iteration, and after every restart, we just take the gradient as direction.
            self.p = -grad
        else:

            #Compute beta according to the variant we are using.
            if self.beta_f == "FR":
                beta = dir_der(grad,grad)/(dir_der(self.last_g,self.last_g)+1e-7)
            if self.beta_f == "PR":
                beta = max(
                    0,dir_der(grad,(grad-self.last_g))/(dir_der(self.last_g,self.last_g)+1e-7))
            self.p = -grad + beta*self.p

        self.last_g = grad

        if self.ls!=None:
            #Perform a line search
            dir_norm=dir_der(-self.p,grad)
            actual_lr = self.ls.search(f, W,loss, self.p, dir_norm)
        else:
            actual_lr = self.lr

        return (W+actual_lr*self.p)


'''
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


    def pprint(self):

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
'''

optimizers = dict()
optimizers["SGD"] = SGD()
optimizers["momentum"] = Momentum()
optimizers["adine"] = Adine()
optimizers["adam"] = Adam()
optimizers["adamax"] = Adamax()
optimizers["rmsprop"] = RMSProp()
optimizers["ConjugateGradient"] = ConjugateGradient()
#optimizers["BFGS"] = BFGS()
