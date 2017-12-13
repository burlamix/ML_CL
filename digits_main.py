import sys
from sklearn import preprocessing
from preproc import *
from NN import *
from optimizer import *
from validation import *
from keras.datasets import mnist
from keras.utils import np_utils


(x_train,y_train) ,(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],784).astype('float32')
y_train = np_utils.to_categorical(y_train,10).astype('float32')
x_test = x_test.reshape(x_test.shape[0],784)
dataset = Dataset()
dataset.init_train((x_train,y_train))
dataset.init_test((x_test,y_test))

NN = NeuralNetwork()
NN.addLayer(784,10,activation="sigmoid")
NN.addLayer(10,10,activation="sigmoid")

opt = SimpleOptimizer(lr=0.01)
NN.fit_ds(dataset, 100, optimizer=opt,batch_size=1000,loss_func='mse', verbose=2)