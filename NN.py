import numpy as np
from preproc import *

class Layer:

    def __init__(self, inputs, neurons, activation, weights=np.array(None), bias=0):
        self.currentOutput = None
        self.grad=None
        if inputs<0 or neurons<0: sys.exit("Expected positive value")
        if weights.any()==None:
            self.W = np.random.rand(neurons, inputs+1)
            #self.W = np.ones((neurons, inputs+1))
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


class NeuralNetwork:

    def __init__(self):
        #TODO initial layers
        self.layers = []

    def addLayer(self,inputs, neurons, activation, weights=np.array(None), bias=0):
        self.layers.append(Layer(inputs, neurons, activation, weights, bias=bias))

    def FP(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.getOutput(x)
        return x

    def BP(self, prediction, real, x_in):
        gradients = []
        loss_func = np.sum(1/(2*prediction.shape[0]) * ((real - prediction) * \
                    (real - prediction)))

        #print(loss_func)
        for i in range(len(self.layers)-1, -1, -1):

            logi = self.layers[i].activation.dxf(self.layers[i].currentOutput)

            #logi = self.layers[i].currentOutput * (1 - self.layers[i].currentOutput)

            if i==(len(self.layers)-1):
                err = logi*(prediction - real)
            else:
                err=np.dot(err,self.layers[i+1].W[:,1:])*logi #error is derivative of activation
                #at current layer * (weights*error at next layer)
            if i==0:
                curro = x_in
            else:
                curro = self.layers[i-1].currentOutput
            curro = np.concatenate((np.ones((curro.shape[0], 1)), curro), axis=1)
            grad = np.dot(curro.transpose(),err) #TODO save gradient in layer
            self.layers[i].grad = grad
            gradients.append(grad)
        return loss_func, np.array(gradients)


    def f(self, chunk):
        def g(W):
            return self.BP(self.FP(chunk), self.real, chunk)
        return g

    def fit(self, dataset, epochs, optimizer, batch_size=-1):

        if batch_size<0:                        #TODO more check
            batch_size=len(dataset.train[0])

        self.real = dataset.train[1]

        for i in range(0, epochs):
            for chunk in range(0,len(dataset.train[0]),batch_size):
                cap = min([len(dataset.train[0]), chunk + batch_size])

                update = optimizer.optimize(self.f(dataset.train[0][chunk:cap]), "ciao")

                for i in range (0,len(self.layers)):
                    self.layers[i].W = self.layers[i].W+update[-i-1].transpose()

  #  def w_update(update):
   #     for i in range (0,len(self.layers)):
    #        self.layers[i].W = self.layers[i].W+update[-i-1].transpose()
