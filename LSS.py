from NN_lib import optimizers
import numpy as np
from NN_lib import linesearches
from test_functions import rosenbrock, matyas_fun
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.mlab import bivariate_normal


def optimize_fun(fun,start,opt,epochs=100):
    '''
    Optimizes the given function with the given parameters and returns the history
    of iterations.
    :param fun: The function to optimizes: must be of the form f(W, only_fp)
    where W is the 2-arguments input matrix (e.g. np.array([[1,2]])) denoting
    the point to evaluate the function at and only_fp is a parameter that
    determines whether we want both the function value and the gradient at W
    or only the value. f must return the value only if only_fp is set to True
    and both the value and gradient (value,gradient) if only_fp is set to False.
    :param start: Denoattes the starting point to begin optimizing the function from
    :param opt: An object denoting the optimizer to utilize for the optimization of f.
    :param epochs: The number of epochs to optimize f over.
    :return: A dictionary of entries 'x', 'y', 'z', 'opt' denoting respectively
    the history of x coordinates, y coordinates, the corresponding function value
    over the iterations and the input optimizer object.
    '''
    W = start
    x = [W[0][0]]
    y = [W[0][1]]
    z = [fun(np.array([[x[0], y[0]]]), only_fp=True)]

    for i in range(0, epochs):
        W = opt.optimize(fun, W)
        x.append(W[0][0])
        y.append(W[0][1])
        newv = fun(W, only_fp=True)
        z.append(newv)
        print('Function value:', newv)

    return {'x':np.array(x),'y':np.array(y),'z':np.array(z),'opt':opt}


def plot_contours(fun, xrange, yrange, contours=200):
    '''
    Plots the specified number of contours of the given input fun, over the given range.
    :param fun: A function of the format specified in @optimize_fun
    :param xrange: The range of x values to plot the contours over. Note that length
    of xrange and yrange must match.
    :param yrange: The range of y values to plot the contours over. Note that length
    of xrange and yrange must match.
    :param contours: The number of contours to display
    :return: The resulting plot handler (use it it @navigate_fun
    '''
    fig = plt.figure()
    # ax = fig.gca(projection='3d')optimize_fun
    ax = fig.gca()
    X, Y = np.meshgrid(xrange, yrange)
    zs = np.array([fun([[xrange, yrange]], only_fp=True)
                   for xrange, yrange in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # surf=ax.plot_surface(Y,X,Z, cmap=cm.jet,cstride=1,rstride=1,alpha=0.3)
    # surf = ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    surf = ax.contour(X, Y, Z, contours, lw=3, colors="k", linestyles="solid")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z Label')
    ax.invert_xaxis()
    return ax


def navigate_fun(navs, plot):
    '''
    Given an array of histories of evaluations (see @optimize_fun return value) and
    an appropriate contours plot (see @plot_contours) displays the interactive
    navigation of the optimizer contained in navs epoch by epoch.
    :param navs: An array of dictionaries formatted as specified in the return value
    of @optimize_fun.
    :param plot: Contour plot handler as specified in @plot_contours
    '''

    #plt.switch_backend('QT4Agg')
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.ion()
    cols = [np.random.rand(3,1) for i in range(0,len(navs))]
    for i in range(0, len(navs[0]['x']) - 1):
        for j in range(0,len(navs)):
            px = [navs[j]['x'][i], navs[j]['x'][i + 1]]
            py = [navs[j]['y'][i], navs[j]['y'][i + 1]]
            pz = [navs[j]['z'][i], navs[j]['z'][i + 1]]
            plot.plot(px, py, 'k', alpha=0.8, linewidth=1.5,
                    c=cols[j],label=navs[j]['opt'].pprint())
            if (i==0): plt.legend()
        plt.pause(.001)
    plt.pause(2**31)
    #input("..")


amg = linesearches.armj_wolfe(m1=1e-4, m2=0.9, lr=0.2, min_lr=1e-11, scale_r=0.9, max_iter=200)
bt = linesearches.back_track(lr=1, m1=1e-4, scale_r=0.9, min_lr=1e-11, max_iter=200)

lso = optimizers.RMSProp(lr=0.0001)
lso = optimizers.Momentum(lr=0.00001)
lso = optimizers.Adine(lr=0.01, ms=0.9, mg=1.0001, e=1.0, ls = None)
lso = optimizers.Momentum(lr=0.01,eps=0.6)
lso = optimizers.Adam(lr=0.1)
lso = optimizers.Momentum(lr=0.1,eps=0.6)
lso = optimizers.SimpleOptimizer(lr=0.1,ls= None)

fun = matyas_fun
a = np.arange(-4, 5, 0.1)
b = np.arange(6, -3, -0.1)
plot = plot_contours(fun, xrange=a, yrange=b, contours=200)

epochs=500
lso = optimizers.ConjugateGradient(lr=0.076,ls=amg)
o1 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.ConjugateGradient(lr=0.076,ls=bt)
o2 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.Adam(lr=0.076)
#o3 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.Adamax(lr=0.076)
#o4 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)

navigate_fun([o1,o2], plot=plot)

