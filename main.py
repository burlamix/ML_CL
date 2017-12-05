import csv
import numpy as np
import sys
from sklearn import preprocessing

train_data_path = "ML-CUP17-TR.csv"
test_data_path = "ML-CUP17-TS.csv"

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

class SimpleOptimizer:

    def optimize(self, epochs, NN, maxIters):
        NN.fp
        NN.bp
        #update weight
        #for layer in NN.layers:
            #layer.weights = layer.weights+layer.gradients

#preprocessing
'''class Layer:

class NeuralNetwork:

    fit():
        #gradient calculation is the same for all optimizers,
        #the update rule changes for each one
        #while backprop set layer.gradient
        self.optimizer.optimize(self.loss_func(dataset.train))

NN = NeuralNetwork()
NN.addLayer(type,regularizer,neurons,activation..)

NN.addLayers(type,[array of neurons], <number of layers>,
                                [array of regularizers],[array of activations])
NN.compile(loss_func, optimizer,(metric)..)

NN.fit(dataset, batch_size, epochs, cvfolds=0, vsplit=0) #only one of cvfolds, vsplit
#grid search function
NN.test()'''

