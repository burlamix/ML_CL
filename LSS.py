from NN_lib import optimizers
import numpy as np
from NN_lib import linesearches
from test_functions import rosenbrock, matyas_fun, himmelblau, simple_fun
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.mlab import bivariate_normal

def optimize_fun(fun,start,opt,epochs=100, min_f=None, min_r=None):
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
    :param min_f: minumun of the function.
    :param min_r range within the minimun to stop at.
    :return: A dictionary of entries 'x', 'y', 'z', 'opt' denoting respectively
    the history of x coordinates, y coordinates, the corresponding function value
    over the iterations and the input optimizer object.
    '''
    W = start
    x = [W[0][0]]
    y = [W[0][1]]
    z = [fun(np.array([[x[0], y[0]]]), only_fp=True)]
    epochs_stop=None

    for i in range(0, epochs):
        W = opt.optimize(fun, W)
        x.append(W[0][0])
        y.append(W[0][1])
        newv = fun(W, only_fp=True)

        if(epochs_stop==None and newv<min_f+min_r and newv > min_f-min_r):
            epochs_stop=i+1

        z.append(newv)
        print('Function value:', newv)

    return {'x':np.array(x),'y':np.array(y),'z':np.array(z),'opt':opt,'epochs_stop':epochs_stop}


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
    #ax = fig.gca(projection='3d') #Uncomment for 3d plot
    ax = fig.gca()
    X, Y = np.meshgrid(xrange, yrange)
    zs = np.array([fun([[xrange, yrange]], only_fp=True)
                   for xrange, yrange in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # surf=ax.plot_surface(Y,X,Z, cmap=cm.jet,cstride=1,rstride=1,alpha=0.3) #
    # surf = ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    surf = ax.contour(X, Y, Z, contours, lw=3, colors="k", linestyles="solid")
    #fig.colorbar(surf, shrink=0.5, aspect=5) #Uncomment for colorbar - useless for contours
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlabel('Z Label') #Uncomment for 3d plot
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
    ann = []
    cols = [np.random.rand(3,1) for i in range(0,len(navs))]
    for i in range(0, len(navs[0]['x']) - 1):
        for j in range(0,len(navs)):
            px = [navs[j]['x'][i], navs[j]['x'][i + 1]]
            py = [navs[j]['y'][i], navs[j]['y'][i + 1]]
            pz = [navs[j]['z'][i], navs[j]['z'][i + 1]]
            es = "no" if navs[j]['epochs_stop']==None else str(navs[j]['epochs_stop'])
            plot.plot(px, py, 'k', alpha=0.8, linewidth=1.5,
                    c=cols[j],label=navs[j]['opt'].pprint()+ ',ce:'+es)
            bbox_args = dict(boxstyle="round", fc="0.8")
            arrow_args = dict(arrowstyle="->")
            ann.append( plot.annotate(str(pz[1]),
                        xy=(px[1], py[1]),  # theta, radius
                        xytext=(px[1]-0.5, py[1]-0.5),  # theta, radius
                        #xycoords='polar',
                        #textcoords='polar',
                        arrowprops=dict(facecolor=cols[j], shrink=0.305),
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        clip_on=True)  # clip to the axes bounding box
                            )
            if (i==0):
                plt.legend(bbox_to_anchor=(1,0), loc="lower right",prop={'size': 10.5})

        plt.pause(.001)
        for a in ann:a.remove()
        ann.clear()

    #input("..")
    plt.pause(2**31) #Might be problematic on some systems --
    # substitute with the commented line above in that case

#Define some line searches
amg = linesearches.armj_wolfe(m1=1e-4, m2=0.9, lr=0.076,min_lr=1e-11, scale_r=0.95, max_iter=100)
bt = linesearches.back_track(lr=2.1, m1=1e-4, scale_r=0.4, min_lr=1e-11, max_iter=200)


fun = himmelblau #Available functions = {matyas_fun, rosenbrock, himmelblau, simple_fun}
x = np.arange(-10, 10, 0.1) #x-range for the plot
y = np.arange(-10, 10, 0.1) #y-range for the plot
plot = plot_contours(fun, xrange=x, yrange=y, contours=200)

min_r=1e-4 #Range within the optimum to stop at
iterations=50 #Maximum number of iterations

#See optimizers for a comprehensive list of the available optimzers
opts = list()
start = np.array([[-8, 2]])
step = 0.0007
opts.append((optimizers.ConjugateGradient(lr=step,ls=amg,restart=2), start))
opts.append((optimizers.ConjugateGradient(lr=step,ls=amg,restart=-1,beta_f="PR"), start))
opts.append((optimizers.Momentum(lr=step, eps=0.9,ls=None), start))
opts.append((optimizers.ConjugateGradient(lr=step,ls=None), start))
opts.append((optimizers.Adam(lr=step,ls=None), start))
opts.append((optimizers.RMSProp(lr=step), start))
opts.append((optimizers.Adine(lr=step), start))
opts.append((optimizers.Adamax(lr=step), start))
opts.append((optimizers.SimpleOptimizer(lr=step,ls=bt), start))


res = [optimize_fun(
    fun, start=o[1], opt=o[0], epochs=iterations, min_f=0, min_r=min_r) for o in opts[0:]]



navigate_fun(res[0:], plot=plot)
