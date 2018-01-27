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
        print('final', newv)

    return {'x':np.array(x),'y':np.array(y),'z':np.array(z),'opt':opt}

def navigate_fun(navs, plot):
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

def plot_contours(fun, xrange, yrange, contours=200):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
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

amg = linesearches.armj_wolfe(m1=1e-4, m2=0.9, lr=0.001, min_lr=1e-11, scale_r=0.5, max_iter=100)
bt = linesearches.back_track(lr=0.15, m1=1e-4, scale_r=0.95, min_lr=1e-11, max_iter=200)

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

epochs=300
lso = optimizers.Adam(lr=0.076,ls=amg)
o1 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.SimpleOptimizer(lr=0.076,ls=amg)
o2 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.Adam(lr=0.076)
#o3 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)
lso = optimizers.Adamax(lr=0.076)
#o4 = optimize_fun(fun,np.array([[-3, 5]]),opt=lso, epochs=epochs)

navigate_fun([o1,o2], plot=plot)

