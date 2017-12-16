import numpy as np
from loss_functions import reguls
from loss_functions import mse, msedx
from optimizer import activations
import types
import sys

class Layer:

    def __init__(self, inputs, neurons, activation, weights=np.array(None), bias=0,
                 regularizer="L2", rlambda = 0.00): #TODO find proper default value for lambda

        if isinstance(regularizer[0], types.FunctionType) and \
            isinstance(regularizer[1], types.FunctionType):
            self.regularizer = regularizer
        else:
            #Otherwise check whether the specified regularizer function exists
            try: self.regularizer = reguls[regularizer]
            except KeyError: sys.exit("regularizer function undefined")


        if isinstance(activation[0], types.FunctionType) and \
            isinstance(activation[1], types.FunctionType):
            self.activation = activation
        else:
            #Otherwise check whether the specified activation function exists
            try: self.activation = activations[activation]
            except KeyError: sys.exit("activation function undefined")




        self.rlambda=rlambda
        self.currentOutput = None
        self.grad=None
        if inputs<0 or neurons<0: sys.exit("Expected positive value")
        self.neurons=neurons
        self.inputs=inputs
        if weights.any()==None:
            self.initialize_random_weights()
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



    def getOutput(self,x):
        
        x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        #print("x------",x)
        #print("W------1x",self.W.transpose())
        #print("W------1x",self.W.transpose().shape)
        #print("x------1x",x.shape)
        partial = np.dot(x, self.W.transpose())
        self.currentOutput = self.activation.f(partial)
        #print("------------------------------------------")
        return self.currentOutput

    def regularize(self):
        return self.regularizer[0](self.W, self.rlambda)

    def regularizedx(self):
        return self.regularizer[1](self.W, self.rlambda)

    def set_weights(self,W):
        self.W=W

    def initialize_random_weights(self, method='xavier'):
        if method == 'xavier':
            var = 2 / (self.inputs + self.neurons)
            self.W = np.random.normal(0, var, (self.neurons, self.inputs+1))
        elif method == 'fan_in':
            self.W =  np.random.uniform(-1 / np.sqrt(self.inputs+1), 1/np.sqrt(self.inputs+1),(self.neurons, self.inputs+1))
        self.W[:,0] = 0
