import csv
import numpy as np

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
        pass
        #split train into k sub-arrays

    def split_train_percent(self, percent):
        pass
        #split train into 2 according to percent

class Preprocessor:

    #apply_normalization(dataset, method="min-max"):
        #normalize training and test data

    def apply_sorting(dataset):
        pass
        #sort dataset train randomly

    def get_means(self):
        pass

    def get_variance(self):
        pass

    def remove_outliers(self, sensitivity=3.0):
        pass

def load_data(path=None, target=True):
    data = []
    if path==None:
        print("Path not specified")
        exit(1) #to change
    with open(path,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = data[10::]
    x = [d[:-2] for d in data]
    if target:
        y = [d[-2:] for d in data]
        return (np.array(x), np.array(y))
    else:
        return (np.array(x), None)

dataset = Dataset()
dataset.init_train(load_data(train_data_path, True))
dataset.init_test(load_data(test_data_path, False))

preprocessor = Preprocessor()

#preprocessing
class Layer:

class NeuralNetwork:



NN = NeuralNetwork()
NN.addLayer(type,regularizer,neurons,activation..)

NN.addLayers(type,[array of neurons], <number of layers>,
                                [array of regularizers],[array of activations])
NN.compile(loss_func, optimizer,(metric)..)

NN.fit(dataset, batch_size, epochs, cvfolds=0, vsplit=0) #only one of cvfolds, vsplit

NN.test()

