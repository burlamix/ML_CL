import csv
import numpy as np
import sys
from sklearn import preprocessing

train_data_path = "ML-CUP17-TR.csv"
test_data_path = "ML-CUP17-TS.csv"
np.random.seed(4)

class Dataset:
    def __init__(self):
        self.train = None
        self.test = None

    def init_train(self, train):
        self.train = train

    def init_test(self, test):
        self.test = test

    def split_train_k(self, k):
        '''Splits training data into k equal or near-equal subarrays
        result is returned as a tuple where the first element is an array containing
        the input data subarrays and the second element is an array containing the
        target data subarrays'''
        input = np.array_split(self.train[0],k)
        target = np.array_split(self.train[1],k)
        return (input, target)

    def split_train_percent(self, percent):
        '''Splits training data into 2 according to percent. Result is returned as a tuple
        where the first element is an array containing the input data subarrays
        and the second element is an array containing the target data subarrays'''
        l = int(len(dataset.train[0])*percent/100)
        return (np.array([dataset.train[0][0:l],dataset.train[0][l::]]),
                np.array([dataset.train[1][0:l], dataset.train[1][l::]]))

class Preprocessor:

    def normalize(self, dataset, method='min-max', norm_output=False):
        if method=='min-max':
            dataset.train[0] =  (dataset.train[0] - dataset.train[0].min(axis=0)) / \
                               (dataset.train[0].max(axis=0) - dataset.train[0].min(axis=0))
            if norm_output:
                dataset.train[1] = (dataset.train[1] - dataset.train[1].min(axis=0)) / \
                               (dataset.train[1].max(axis=0) - dataset.train[1].min(axis=0))
        #normalize training and test data

    def shuffle(self,dataset):
        '''randomly shuffles training data'''
        perm = np.random.permutation(len(dataset.train[0]))
        dataset.train = [dataset.train[0][perm], dataset.train[1][perm]]

    def get_means(self):
        pass

    def get_variance(self):
        pass

    def remove_outliers(self, sensitivity=3.0):
        pass

def load_data(path=None, target=True, header_l=0, targets=0):
    '''Loads data into numpy arrays from given path. File at specified path
    must be in CSV format. If target output is in the file it is assumed to occupy
    the last targets columns.

    Attributes:
        path: the path of the CSV file containing the data
        target: set to true to indicate CSV contains target outputs
        header_l: specifies the number of header rows
        targets: specifies the number of targets
        '''
    data = []
    if path==None:
        sys.exit("Path not specified")
    with open(path,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = data[header_l::]
    if target:
        x = [d[:-targets] for d in data]
        y = [d[-targets:] for d in data]
        return [np.array(x).astype('float32'), np.array(y).astype('float32')]
    else:
        x = [d for d in data]
        return [np.array(x).astype('float32'), None]

dataset = Dataset()
dataset.init_train(load_data(train_data_path, True, header_l=10, targets=2))
dataset.init_test(load_data(test_data_path, False, header_l=10))

preprocessor = Preprocessor()

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Activation:

    def __init__(self,f,dxf,gradf):
        self.f = f
        self.dxf = dxf

class SimpleOptimizer:

    def optimize(self, y_true, epochs, NN, maxIters):
        for i in range(epochs):
            o = NN.fp
            NN.bp(o, y_true)
            #update weight
            #for layer in NN.layers:
                #layer.weights = layer.weights+layer.gradients

#preprocessing
class Layer:

    def __init__(self, inputs, neurons, activation, weights=None, bias=0):
        self.currentOutput = None
        if inputs<0 or neurons<0: sys.exit("Expected positive value")
        if weights.any()==None:
            self.W = np.random.rand(neurons, inputs+1)
            self.W = np.ones((neurons, inputs+1))
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
        print(x)
        print(self.W.transpose())
        partial = np.dot(x, self.W.transpose())
        self.currentOutput = self.activation.f(partial)
        return self.currentOutput


class NeuralNetwork:

    def __init__(self):
        #TODO initial layers
        self.layers = []

    def addLayer(self,inputs, neurons, activation, weights=None, bias=0):
        self.layers.append(Layer(inputs, neurons, activation, weights, bias=bias))

    def FP(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.getOutput(x)
        return x

    def BP(self, prediction, real, x_in):
        for i in range(len(self.layers)-1, -1, -1):
            loss_func = 1 / 2 * (real - self.layers[i].currentOutput) * \
                        (real - self.layers[i].currentOutput)
            der_loss_func = (prediction - real)
            logi = self.layers[i].currentOutput * (1 - self.layers[i].currentOutput)
            if i==(len(self.layers)-1):err = logi*der_loss_func
            else:
                err=np.dot(err,self.layers[i+1].W[:,1:])
            if i==0:curro = x_in
            else:
                curro = self.layers[i-1].currentOutput
            grad = curro*err*logi #TODO save gradient in layer


    #def fit():
        #gradient calculation is the same for all optimizers,
        #the update rule changes for each one
        #while backprop set layer.gradient
     #   self.optimizer.optimize(self.loss_func(dataset.train))


toyx = np.asarray([[0.05,0.1]]) #TODO make it work with arrays
activation = Activation(sigmoid,sigmoid,sigmoid)
NN = NeuralNetwork()
NN.addLayer(2,2,activation,weights=np.array([[0.15,0.20],[0.25,0.3]]),bias=0.35)
NN.addLayer(2,2,activation,weights=np.array([[0.40,0.45],[0.50,0.55]]),bias=0.6)
o = NN.FP(toyx)
print(o)
print("==========================")
o1 = NN.BP(o,np.array([0.01,0.99]),toyx)
print(o1)
'''NN = NeuralNetwork()
NN.addLayer(type,regularizer,neurons,activation..)

NN.addLayers(type,[array of neurons], <number of layers>,
                                [array of regularizers],[array of activations])
NN.compile(loss_func, optimizer,(metric)..)

NN.fit(dataset, batch_size, epochs, cvfolds=0, vsplit=0) #only one of cvfolds, vsplit
#grid search function
NN.test()'''

