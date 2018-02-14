import numpy as np



def dir_der(grad,dir):
    '''
    Directional derivative, i.e. <grad, dir>
    '''
    if(len((grad.shape)) == 2 ):
        a = grad.reshape(grad.shape[1] * grad.shape[0])
        b = dir.reshape(dir.shape[1] * dir.shape[0])
        return np.sum(a * b)
    if (grad[0].shape == ()):
        return np.sum(grad * dir)
    else:
        a = np.concatenate(
            [grad[i].reshape(grad[i].shape[1] * grad1[i].shape[0], 1) for i in range(0, len(grad))])
        b = np.concatenate(
            [dir[i].reshape(dir[i].shape[1] * dir[i].shape[0], 1) for i in range(0, len(dir))])
        return np.sum(a * b)

class BackTracking():
    '''
    Backtracking line search.
    '''
    def __eq__(self, other):
        if (other is None): return False
        return self.lr==other.lr \
               and self.m1==other.m1 and self.scale_r==other.scale_r and \
                    self.max_iter == other.max_iter and\
                        self.min_lr.__eq__(other.min_lr)

    def pprint(self):
        return "BT{lr:" + str(self.lr) + ",m1:" + str(self.m1) +\
            ",r:" + str(self.scale_r) + ",i:" + str(self.max_iter)

    def __init__(self,lr=1, m1=1e-4, scale_r=0.4, min_lr=1e-11, max_iter=100):
        '''
        :param lr: Initial step size. Gonna backtrack from this.
        :param m1: Parameter of the Armijo condition
        :param scale_r: Scale factor to backtrack by.
        :param min_lr: Minimum step size.
        :param max_iter: Max number of iterations before giving up.
        '''
        self.lr = lr
        self.m1 = m1
        self.scale_r = scale_r
        self.min_lr = min_lr
        self.max_iter = max_iter

    def search(self, f, W, curr_v, dir,phip0):
        lr_in = self.lr
        maxiter_in = self.max_iter

        while maxiter_in > 0 and lr_in > self.min_lr:
            phia = f(W + lr_in * dir, only_fp=True) #Check where we would end up
            if phia <= curr_v + self.m1 * lr_in * phip0: #If it satisfies armijo condition, we are done
                return lr_in
            lr_in = lr_in * self.scale_r #Otherwise have to keep looking
            maxiter_in -= 1
        return lr_in


class ArmijoWolfe():
    #Armijo-wolfe line search with strong wolfe condition
    #Based on implementation by Antonio Frangioni
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
        '''
        :param m1: Parameter of the Armijo condition.
        :param m2: Parameter of the Wolfe condition.
        :param lr: Step size to start from.
        :param min_lr: Minimum step size.
        :param scale_r: Scale factor of the step size.
        :param max_iter: Max number of iterations before giving up.
        '''
        self.lr = lr
        self.m1 = m1
        self.m2 = m2
        self.scale_r = scale_r
        self.min_lr = min_lr
        self.max_iter = max_iter

    def search(self, f, W, curr_v, curr_grad, phip0):
        lr_in = self.lr
        max_iter_in = self.max_iter

        while max_iter_in > 0:
            # phia value of the function where we would end up
            # phips_p gradient where we would end up
            phia, phips_p = f(W + lr_in * curr_grad)
            phips = dir_der(curr_grad, phips_p)

            #Test armijo and wolfe conditions
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

        while (max_iter_in > 0) and ((lr_in - am) > self.min_lr and phips > 1e-12):
            #Interpolate
            a = (am * phips - lr_in * phipm) / (phips - phipm)
            temp = min([lr_in * (1 - sf), a])
            a = max([am * (1 + sf), temp])
            phia, phipp = f(W + a * curr_grad)
            phip = dir_der(curr_grad, phipp)
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