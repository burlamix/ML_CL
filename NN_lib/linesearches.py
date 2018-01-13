import numpy as np


#Needs fixing
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

#Prolly not needed..
def us_norm(x):
    if (x[0].shape == ()):
        return -np.linalg.norm(x)
    else:
        return -(np.linalg.norm(
            np.concatenate(
                [x[i].reshape(x[i].shape[1] * x[i].shape[0], 1) for i in range(0, len(x))])
        ))

def dir_der(grad, lastg):
    if (grad[0].shape==()):
        return grad.transpose()*lastg #Didn't even check this
    a = np.concatenate(
        [grad[i].reshape(grad[i].shape[1] * grad[i].shape[0], 1) for i in range(0, len(grad))])
    b = np.concatenate(
        [lastg[i].reshape(lastg[i].shape[1] * lastg[i].shape[0], 1) for i in range(0, len(lastg))])
    return np.sum(-a * b)

def armj_wolfe(m1=1e-4, m2=0.9, lr=0.1, min_lr=1e-11, scale_r=0.9, max_iter=100):
    #Armijo-wolfe line search with strong wolfe condition
    #Credits to Antonio Frangioni
    def armj_wolfe_internal(f, W, curr_v, curr_grad):
        lr_in = lr
        max_iter_in = max_iter
        # phip0- directional derivative -> use norm of current gradient
        phip0 = dir_der(curr_grad, curr_grad)
        while max_iter_in > 0:
            # phia value of the function where we would go
            # phips_p gradient where we would go
            phia, phips_p = f(W - lr_in * curr_grad)
            phips = dir_der(curr_grad, phips_p)
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
            phia, phipp = f(W - a * curr_grad)
            phip = dir_der(curr_grad, phipp)
            if ((phia <= curr_v + m1 * a * phip0) and (np.abs(phip) <= -m2 * (phip0))):
                return a
            if phip < 0:
                am = a
                phipm = phip
            else:
                lr_in = a
                if lr_in < min_lr:
                    return lr_in
                phips = phip
                max_iter_in -= 1
        return a

    return armj_wolfe_internal

def back_track(lr=1, m1=1e-4, scale_r=0.1, min_lr=1e-11, max_iter=0):

    def back_track_internal(f, W, curr_v, curr_grad):
        lr_in = lr
        maxiter_in = max_iter

        if (curr_grad[0].shape == ()):
            phip0 = -np.linalg.norm(curr_grad)
        else:
            phip0 = us_norm(curr_grad)

        while maxiter_in > 0 and lr_in > min_lr:
            phia = f(W - lr_in * curr_grad, only_fp=True)
            if phia <= curr_v + m1 * lr_in * phip0:
                return lr_in
            lr_in = lr_in * scale_r
            maxiter_in -= 1
        return lr_in
    return back_track_internal
