from keras.datasets import mnist
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from NN_lib.NN import *
from NN_lib.optimizers import *
from NN_lib import preproc

#Loading the dataset
(x_train,y_train) ,(x_test, y_test) = mnist.load_data()

#reshaping data
x_train = x_train.reshape(x_train.shape[0],784).astype('float32')
y_train = np_utils.to_categorical(y_train,10).astype('float32')
x_test = x_test.reshape(x_test.shape[0],784)
y_test = np_utils.to_categorical(y_test,10).astype('float32')

#inizialization of dataset object for a easier manipulation
dataset = preproc.Dataset()
dataset.init_train((x_train,y_train))
dataset.init_test((x_test,y_test))

#bulding the NN with our library
NN = NeuralNetwork()
NN.addLayer(784,10,activation="linear")
NN.addLayer(10,10,activation="softmax")

#bulding the optimizer
opt = SGD(lr=0.01)

#preprocessing and training of the neural network
pr = preproc.Preprocessor()
pr.shuffle(dataset)
NN.fit_ds(dataset, 100, optimizer=opt,batch_size=32,loss_func='mse', verbose=2)


#inizialize initial keras matrix weight
ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/20), seed=None)
ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/4), seed=None)

#buld keras network
model = Sequential()
model.add(Dense(10, activation= 'linear' ,kernel_initializer=ini1,input_dim=784,use_bias=False))
model.add(Dense(10, activation= 'softmax',kernel_initializer=ini2 ,use_bias=False))



#build keras optimizer
#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = SGD(lr=0.01, momentum=0.0, decay=0.00 )


model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])


#train keras network
model.fit(x_train, y_train,batch_size=32,epochs=10000,shuffle=False)