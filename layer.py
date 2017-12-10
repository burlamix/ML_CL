import numpy as np
from loss_functions import reguls
from loss_functions import mse, msedx
import types

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
