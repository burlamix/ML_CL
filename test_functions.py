import numpy as np

def matyas_fun(W, only_fp=False):
    r = 0.26 * (W[0][0] ** 2 + W[0][1] ** 2) - 0.48 * W[0][0] * W[0][1];

    if only_fp:
        return r
    else:
        return r, np.array(np.array([[0.52 * W[0][0] - 0.48 * W[0][1], 0.52 * W[0][1] - 0.48 * W[0][0]]]))


def rosenbrock(W, only_fp=False):
    r = 100 * ((W[0][1] - W[0][0] ** 2) ** 2) + (W[0][0] - 1) ** 2

    if only_fp:
        return r
    else:
        return r, np.array([[2 * (200 * W[0][0] ** 3 - 200 * W[0][0] * W[0][1] + W[0][0] - 1),
                            200 * (W[0][1] - W[0][0] ** 2)]])

def himmelblau(W, only_fp=False):
    r = ((W[0][0]**2 + W[0][1]-11) ** 2)+((W[0][0] + (W[0][1]**2)-7) ** 2)

    if only_fp:
        return r
    else:
        return r, np.array([[4*W[0][0] * (W[0][0]**2 + W[0][1]-11) +2*(W[0][0] + (W[0][1]**2)-7),
                             4*W[0][1] * (W[0][0] + W[0][1]**2-7)+2*(W[0][0]**2 + (W[0][1])-11) ]])


def simple_fun(W, only_fp=False):
    r = ((W[0][1] + W[0][0]) ** 2)

    if only_fp:
        return r
    else:
        return r, np.array([[2*(W[0][1] + W[0][0]),2*(W[0][1] + W[0][0])]])

def ackley(W, only_fp=False):
    r = -20*np.exp(-0.2*np.sqrt(0.5*(W[0][0]**2+W[0][1]**2)))-np.exp((1/2)*(np.cos(2*np.pi*W[0][0])+np.cos(2*np.pi*W[0][1])))+np.e + 20
    if only_fp:
        return r
    else:
        dx = (2.82843*W[0][0]*np.exp(-0.14121*np.sqrt(W[0][0]**2+W[0][1]**2)))/np.sqrt(W[0][0]**2+W[0][1]**2+1e-7)+\
        np.pi*np.sin(2*np.pi*(W[0][0]))*np.exp((1/2)*(np.cos(2*np.pi*W[0][0])+np.cos(2*np.pi*W[0][1])))
        dy = (2.82843*W[0][1]*np.exp(-0.14121*np.sqrt(W[0][0]**2+W[0][1]**2)))/np.sqrt(W[0][0]**2+W[0][1]**2+1e-7)+\
        np.pi*np.sin(2*np.pi*(W[0][1]))*np.exp((1/2)*(np.cos(2*np.pi*W[0][0])+np.cos(2*np.pi*W[0][1])))

        return r, np.array([[dx,dy]])

def ff_2(W, only_fp=False):
    r  = np.sin(W[0][0])+np.sin(W[0][1])+2
    if only_fp:
        return r
    else:
        return r, np.array([[np.cos(W[0][0]),np.cos(W[0][1])]])


def easom(W, only_fp=False):
    r = -np.cos(W[0][0])*np.cos(W[0][1])*np.exp(-(W[0][0]-np.pi)**2-(W[0][1]-np.pi)**2)+1
    if only_fp:
        return r
    else:
        dx = np.exp(-(W[0][0]-np.pi)**2-(W[0][1]-np.pi)**2)*np.cos(W[0][1])*(np.sin(W[0][0])+2*(W[0][0]-np.pi)*np.cos(W[0][0]))
        dy = np.exp(-(W[0][0]-np.pi)**2-(W[0][1]-np.pi)**2)*np.cos(W[0][0])*(np.sin(W[0][1])+2*(W[0][1]-np.pi)*np.cos(W[0][1]))
        return r, np.array([[dx,dy]])