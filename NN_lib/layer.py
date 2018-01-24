import numpy as np
from NN_lib import regularizations, activations
import sys


class Layer:

    def __init__(self, inputs, neurons, activation, weights=np.array(None), bias=0, weights_init='fan_in',
                 regularizer="L2", rlambda = 0.00, dropout=0.0):
        '''
        @:NN_lib:NeuralNetwork.addLayer
        '''
        self.regularizer = regularizations.validate_regularizer(regularizer)
        self.activation = activations.validate_activation(activation)
        self.rlambda = rlambda
        self.dropout = dropout
        self.mask = np.ones(neurons)
        self.currentOutput = None
        self.grad=None
        if inputs<0 or neurons<0: sys.exit("Expected positive value")
        self.neurons=neurons
        self.inputs=inputs

        #If no weights were specified, initialized them randomly based
        #on the specified or default method.
        if weights.any()==None:
            self.initialize_random_weights(weights_init)
        else:
            #If a weights matrix has been specified, check the dimensions
            #and initalize the layer accorindgly
            if isinstance(weights, (np.ndarray, np.generic)):
                if (weights.shape!=(neurons, inputs)):
                    sys.exit("Weights wrong dimension")
                else:
                    self.W = weights
                    self.W = np.concatenate((np.ones((self.W.shape[0], 1)) , self.W), axis=1)
            else:
                sys.exit("Expected a nparray")

        self.W[:, 0] = bias

    def getOutput(self,x):
        '''
        Calculate the output of the layer
        :param x: Input of the layer
        :return:
        '''
        x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        print('x',x.shape)
        print('W',self.W.shape)
        partial = np.dot(x, self.W.transpose())
        print('partial',partial.shape)
        self.currentOutput = self.activation.f(partial)
        #If a dropout value has been specified, apply inverted dropout
        #https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/
        if self.dropout !=0:
            self.mask = np.random.binomial(1,1-self.dropout,self.currentOutput.shape)
            self.currentOutput = self.currentOutput*self.mask
            if self.dropout==1:self.mask=np.zeros_like(self.currentOutput)
            else:self.mask= self.mask/(1-self.dropout)
            self.mask = self.mask / (1 - self.dropout if self.dropout != 1 else 1)
        return self.currentOutput

    def regularize(self):
        #Calculate the regularization on the layer
        return self.regularizer.f(self.W, self.rlambda)

    def regularizedx(self):
        #Calculate the derivative of the regularization on the layer
        return self.regularizer.dxf(self.W, self.rlambda)

    def set_weights(self,W):
        #Set the weights of the layer
        self.W=W.transpose()

    def initialize_random_weights(self, weights_init='fan_in'):
        #Randomly initialize the weights of the layer
        #according to the specified method
        if weights_init == 'xavier':
            var = 2 / (self.inputs + self.neurons)
            self.W = np.random.normal(0, var, (self.neurons, self.inputs+1))
        elif weights_init == 'fan_in':
            self.W =  np.random.uniform(-0.7 / self.inputs, 0.7/self.inputs,(self.neurons, self.inputs+1))
        self.W[:,0] = 0
