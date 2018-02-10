import numpy as np


#Needs fixing
def bisection( f, W,grad,eps):
    a_minus = 0
    a_plus = 10
    max_iter = 10
    while (max_iter>0):
        max_iter-=1
        alpha = np.average([a_minus,a_plus])
        val, v = f(W+alpha*grad)
        if (np.abs(v)<=eps):
            break
        if (v>0):
            a_minus = alpha
        else:
            a_plus = alpha
    return alpha

def us_norm2(grad1,grad2):
    if(len((grad1.shape)) == 2 ):
        a = grad1.reshape(grad1.shape[1] * grad1.shape[0])
        b = grad2.reshape(grad2.shape[1] * grad2.shape[0])
        return np.sum(a * b)
    if (grad1[0].shape == ()):
        return np.linalg.norm(grad1)
    else:
        a = np.concatenate(
            [grad1[i].reshape(grad1[i].shape[1] * grad1[i].shape[0], 1) for i in range(0, len(grad1))])
        b = np.concatenate(
            [grad2[i].reshape(grad2[i].shape[1] * grad2[i].shape[0], 1) for i in range(0, len(grad2))])
        return np.sum(a * b)


#Prolly not needed..
def us_norm(grad):
    if(len((grad.shape)) == 2 ):
        return -np.linalg.norm(grad)
    if (grad[0].shape == ()):
        return -np.linalg.norm(grad)
    else:
        return -(np.linalg.norm(
            np.concatenate(
                [grad[i].reshape(grad[i].shape[1] * grad[i].shape[0], 1) for i in range(0, len(grad))])
        ))

def dir_der(grad, lastg):

    if(len((grad.shape)) == 2 ):
        a = grad.reshape(grad.shape[1] * grad.shape[0])
        b = lastg.reshape(lastg.shape[1] * lastg.shape[0])
        return np.sum(-a * b)
 
    if (grad[0].shape==()):
            return grad.transpose()*lastg #Didn't even check this
    else:
      
        a = np.concatenate(
            [grad[i].reshape(grad[i].shape[1] * grad[i].shape[0], 1) for i in range(0, len(grad))])
        b = np.concatenate(
            [lastg[i].reshape(lastg[i].shape[1] * lastg[i].shape[0], 1) for i in range(0, len(lastg))])

    return np.sum(-a * b)

def armj_wolfe(m1=1e-4, m2=0.9, lr=0.1, min_lr=1e-11, scale_r=0.9, max_iter=100):
    #Armijo-wolfe line search with strong wolfe condition
    #Credits to Antonio Frangioni
    def armj_wolfe_internal(f, W, curr_v,curr_grad,phip0):
        lr_in = lr
        max_iter_in = max_iter
        # phip0- directional derivative -> use norm of current gradient
        #phip0 = dir_der(curr_grad, curr_grad)

        while max_iter_in > 0:
            # phia value of the function where we would go
            # phips_p gradient where we would go
            phia, phips_p = f(W + lr_in * curr_grad)
            phips = us_norm2(curr_grad, phips_p)
            # test armijo strong wolfe condiction

            if phia <= curr_v + m1 * lr_in * phip0 and (np.abs(phips) <= -m2 * phip0):
                return lr_in
            if phips >= 0:
                break
            else:
                lr_in = lr_in / scale_r
                max_iter_in -= 1
        am = 0
        a = lr_in
        sf = 1e-3
        phipm = phip0
        while (max_iter_in > 0) and ((lr_in - am) > min_lr and phips > 1e-12):
            a = (am * phips - lr_in * phipm) / (phips - phipm)
            temp = min([lr_in * (1 - sf), a])
            a = max([am * (1 + sf), temp])
            phia, phipp = f(W + a * curr_grad)
            phip = us_norm2(curr_grad, phipp)
            if ((phia <= curr_v + m1 * a * phip0) and (np.abs(phip) <= -m2 * (phip0))):
                return a
            if phip < 0:
                am = a
                phipm = phip
            else:
                lr_in = a
                if lr_in < min_lr:
                    return min_lr
                phips = phip
                max_iter_in -= 1
        return a

    return armj_wolfe_internal

def back_track(lr=1, m1=1e-4, scale_r=0.1, min_lr=1e-11, max_iter=100):

    def back_track_internal(f, W, curr_v, dir,phip0):
        lr_in = lr
        maxiter_in = max_iter

        #if (curr_grad[0].shape == ()):
        #    phip0 = -np.linalg.norm(curr_grad)
        #else:
        #    phip0 = us_norm(curr_grad)

        while maxiter_in > 0 and lr_in > min_lr:
            phia = f(W + lr_in * dir, only_fp=True)
            if phia <= curr_v + m1 * lr_in * phip0:
                return lr_in
            lr_in = lr_in * scale_r
            maxiter_in -= 1
        return lr_in
    return back_track_internal

class BackTracking():

    def __eq__(self, other):
        if (other is None and self is not None): return False
        return self.__str__().__eq__(other.__str__()) and self.lr==other.lr \
               and self.m1==other.m1 and self.scale_r==other.scale_r and \
                    self.max_iter == other.max_iter and\
                        self.min_lr.__eq__(other.min_lr)

    def pprint(self):
        return "BT{lr:" + str(self.lr) + ",m1:" + str(self.m1) + ",m2:" +\
            ",r:" + str(self.scale_r) + ",i:" + str(self.max_iter)

    def __init__(self,lr=1, m1=1e-4, scale_r=0.1, min_lr=1e-11, max_iter=100):
        self.lr = lr
        self.m1 = m1
        self.scale_r = scale_r
        self.min_lr = min_lr
        self.max_iter = max_iter

    def search(self, f, W, curr_v, dir,phip0):
        lr_in = self.lr
        maxiter_in = self.max_iter

        while maxiter_in > 0 and lr_in > self.min_lr:
            phia = f(W + lr_in * dir, only_fp=True)
            if phia <= curr_v + self.m1 * lr_in * phip0:
                return lr_in
            lr_in = lr_in * self.scale_r
            maxiter_in -= 1
        return lr_in


class ArmijoWolfe():
    #Armijo-wolfe line search with strong wolfe condition
    #Credits to Antonio Frangioni

    def __eq__(self, other):
        if (other is None): return False
        return self.lr == other.lr \
               and self.m1 == other.m1 and self.m2 == other.m2 and \
               self.scale_r == other.scale_r and \
               self.max_iter == other.max_iter and \
               self.min_lr.__eq__(other.min_lr)

    def pprint(self):
        return "AW{lr:"+str(self.lr)+",m1:"+str(self.m1)+",m2:"+str(self.m2)+ \
               ",r:" + str(self.scale_r) + ",i:"+str(self.max_iter)

    def __init__(self, m1=1e-4, m2=0.9, lr=0.1, min_lr=1e-11, scale_r=0.9, max_iter=100):
        self.lr = lr
        self.m1 = m1
        self.m2 = m2
        self.scale_r = scale_r
        self.min_lr = min_lr
        self.max_iter = max_iter

    def search(self, f, W, curr_v, curr_grad, phip0):
        lr_in = self.lr
        max_iter_in = self.max_iter
        # phip0- directional derivative -> use norm of current gradient
        # phip0 = dir_der(curr_grad, curr_grad)

        while max_iter_in > 0:
            # phia value of the function where we would go
            # phips_p gradient where we would go
            phia, phips_p = f(W + lr_in * curr_grad)
            phips = us_norm2(curr_grad, phips_p)
            # test armijo strong wolfe condiction

            if phia <= curr_v + self.m1 * lr_in * phip0 and \
                    (np.abs(phips) <= -self.m2 * phip0):
                return lr_in
            if phips >= 0:
                break
            else:
                lr_in = lr_in / self.scale_r
                max_iter_in -= 1
        am = 0
        a = lr_in
        sf = 1e-3
        phipm = phip0
        #input('..')
        while (max_iter_in > 0) and ((lr_in - am) > self.min_lr and phips > 1e-12):
            a = (am * phips - lr_in * phipm) / (phips - phipm)
            temp = min([lr_in * (1 - sf), a])
            a = max([am * (1 + sf), temp])
            #print('1:',am * (1 + sf))
            #print('temp:',temp)
            #print('a:',a)
            #print('**')
            phia, phipp = f(W + a * curr_grad)
            phip = us_norm2(curr_grad, phipp)
            if ((phia <= curr_v + self.m1 * a * phip0) and (np.abs(phip) <= np.abs(self.m2 * (phip0)))):
                return a
            if phip < 0:
                am = a
                phipm = phip
            else:
                lr_in = a
                if lr_in < self.min_lr:
                    return self.min_lr
                phips = phip
            max_iter_in -= 1
        return a