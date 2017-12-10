import numpy as np
import sys
from preproc import *
from loss_functions import mse, msedx
from loss_functions import losses
from loss_functions import reguls

import types

class Layer:

    def __init__(self, inputs, neurons, activation, weights=np.array(None), bias=0,
                 regularizer="L2", rlambda = 0.01): #TODO find proper default value for lambda

        if isinstance(regularizer[0], types.FunctionType) and \
            isinstance(regularizer[1], types.FunctionType):
            self.regularizer = regularizer
        else:
            #Otherwise check whether the specified regularizer function exists
            try: self.regularizer = reguls[regularizer]
            except KeyError: sys.exit("regularizer function undefined")

        self.rlambda=rlambda
        self.currentOutput = None
        self.grad=None
        if inputs<0 or neurons<0: sys.exit("Expected positive value")
        self.neurons=neurons
        self.inputs=inputs
        if weights.any()==None:
            self.initialize_random_weight()
        else:
            if isinstance(weights, (np.ndarray, np.generic)):
                if (weights.shape!=(neurons, inputs)):
                    sys.exit("Weights wrong dimension")
                else:
                    self.W = weights
                    self.W = np.concatenate((np.ones((self.W.shape[0], 1)) * 0, self.W), axis=1)
                    self.W[:, 0] = bias
            else:
                sys.exit("Expected a nparray")
        self.activation = activation #check for errors(wheter its a func or not)

    def getOutput(self,x):
        x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        partial = np.dot(x, self.W.transpose())
        self.currentOutput = self.activation.f(partial)
        return self.currentOutput

    def regularize(self):
        return self.regularizer[0](self.W, self.rlambda)

    def regularizedx(self):
        return self.regularizer[1](self.W, self.rlambda)

    def initialize_random_weight(self):
        self.W = np.random.uniform(-0.5,0.5,(self.neurons, self.inputs+1))
        self.W[:,0] = 0 #TODO PROPER WEIGHTS INITIALIZATION(E.G. Xavier)



class NeuralNetwork:

    def __init__(self):
        #TODO initial layers
        self.layers = []

    def addLayer(self,inputs, neurons, activation, weights=np.array(None), bias=0,
                 regularization="L2", rlambda = 0.0):
        self.layers.append(Layer(inputs, neurons, activation, weights, bias,
                                 regularization, rlambda))

    def FP(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.getOutput(x)
        return x

    def BP(self, prediction, real, x_in):
        gradients = []
        loss_func = self.loss_func[0](real,prediction) #+ self.regul()
        #print("loss:"+str(loss_func))
       # if(loss_func<0.075):exit(1)
        for i in range(len(self.layers)-1, -1, -1):

            logi = self.layers[i].activation.dxf(self.layers[i].currentOutput)

            if i==(len(self.layers)-1):
                #err = np.dot(logi,self.loss_func[1](prediction, real))
                err = logi*self.loss_func[1](prediction, real)
            else:
                err=np.dot(err,self.layers[i+1].W[:,1:])*logi #error is derivative of activation
                #at current layer * (weights*error at next layer)
            if i==0:
                curro = x_in
            else:
                curro = self.layers[i-1].currentOutput
            curro = np.concatenate((np.ones((curro.shape[0], 1)), curro), axis=1)
            grad = np.dot(curro.transpose(),err) #TODO save gradient in layer

            grad = grad/real.size
            self.layers[i].grad = grad
            #print("grad:"+str(err))
            gradients.append(grad)
        return loss_func, np.array(gradients)


    def regul(self):
        regul_loss = 0
        for l in self.layers:
            regul_loss+=l.regularize()
        return regul_loss/len(self.dataset.train[0])

#    def reguldx(self):
#        regul_loss = 0
#        for l in self.layers:
#            regul_loss+=l.regularizedx()
#        return regul_loss/len(self.dataset.train[0])

    def reguldx(self,i):
        return self.layers[i].regularizedx()


    def f(self, in_chunk, out_chunk):
        def g(W):
            return self.BP(self.FP(in_chunk), out_chunk, in_chunk)
        return g

    def fit(self, dataset, epochs, optimizer, batch_size=-1, loss_func="mse"):
        #***GENERAL DESCRIPTION***
        #loss_func: can either be a string refering to a standardized defined
        #loss functions or a tuple where the first element is a loss function
        #and the second element is the corresponding derivative
        #
        #batch_size: the dimension of the samples to use for each update step.
        #note that a higher value leads to higher stability and parallelization
        #capabilities, possibly at the cost of a higher number of updates
        ######################################################################
        self.dataset = dataset
        #Check whether the user provided a properly formatted loss function
        if isinstance(loss_func[0], types.FunctionType) and \
                isinstance(loss_func[1], types.FunctionType):
            self.loss_func = loss_func
        else:
            #Otherwise check whether a the specified loss function exists
            try: self.loss_func = losses[loss_func]
            except KeyError: sys.exit("Loss function undefined")

        if batch_size<0 or batch_size>(len(dataset.train[0])):#TODO more check
            batch_size=len(dataset.train[0])


        for i in range(0, epochs):
            #print(i)
            for chunk in range(0,len(dataset.train[0]),batch_size):
                cap = min([len(dataset.train[0]), chunk + batch_size])

                update = optimizer.optimize(self.f(dataset.train[0][chunk:cap], dataset.train[1][chunk:cap]), "ciao")

                for i in range (0,len(self.layers)):
                    self.layers[i].W = self.layers[i].W+update[-i-1].transpose()- (self.reguldx(i)/batch_size)
                    #self.layers[i].W = self.layers[i].W+update[-i-1].transpose()- (self.reguldx(i))

    #TODO this needs to be on test set not train
    #TODO Also not very handy to use a dataset here..
    def evaluate(self,dataset):

        real = self.FP(self.dataset.train[0])

        #val_loss_func = self.loss_func[0](real,dataset.test[0]) #+ self.regul()        TODO TODO TODO MUST cambiare così 
        val_loss_func = self.loss_func[0](real,dataset.train[1]) #+ self.regul()
        return val_loss_func


    def initialize_random_weight(self):
    # inizialize all weight of all layers
    #                                       Optional set how random inizialize??
        for layer in self.layers:
            layer.initialize_random_weight()

  #  def w_update(update):
   #     for i in range (0,len(self.layers)):
    #        self.layers[i].W = self.layers[i].W+update[-i-1].transpose()
