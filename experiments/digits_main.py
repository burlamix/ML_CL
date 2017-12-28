from keras.datasets import mnist
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from NN_lib.NN import *
from NN_lib.optimizers import *

(x_train,y_train) ,(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],784).astype('float32')
y_train = np_utils.to_categorical(y_train,10).astype('float32')
x_test = x_test.reshape(x_test.shape[0],784)

dataset = Dataset()
dataset.init_train((x_train,y_train))
dataset.init_test((x_test,y_test))

NN = NeuralNetwork()
NN.addLayer(784,10,activation="linear")
NN.addLayer(10,10,activation="softmax")

opt = SimpleOptimizer(lr=0.9)

pr = Preprocessor()
pr.shuffle(dataset)
NN.fit_ds(dataset, 100, optimizer=opt,batch_size=32,loss_func='mse', verbose=2)



ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/20), seed=None)
ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/4), seed=None)


model = Sequential()
model.add(Dense(10, activation= 'linear' ,kernel_initializer=ini1,input_dim=784,use_bias=False))
model.add(Dense(10, activation= 'softmax',kernel_initializer=ini2 ,use_bias=False))




#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = SGD(lr=0.1, momentum=0.0, decay=0.00 )

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

#s=0
#for l in model.layers:
#    s+=np.sum(np.abs(l.get_weights()))
#àprint(np.sum(s))
#np.random.seed(5)


model.fit(x_train, y_train,batch_size=32,epochs=10000,shuffle=False)