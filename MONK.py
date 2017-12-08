import preproc
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from NN import *
from optimizer import *

path = "MONK_data/monks-1.train"

data = []
with open(path, 'r') as f:
    reader = csv.reader(f,delimiter=" ")
    for row in reader:
        data.append(row)

x = [d[2:-1] for d in data]
y = [d[1] for d in data]

np.random.seed(5)
x = np.array(x).astype('float32')
y = np.array(y).astype('float32')
y = y.reshape((y.shape[0],1))
optimizer = SimpleOptimizer(lr=0.06)
activation = Activation(sigmoid,sigmoddxf,sigmoid)
activation1 = Activation(linear,lineardxf,sigmoid)
dataset = preproc.Dataset()
dataset.init_train([x,y])
NN = NeuralNetwork()
#NN.addLayer(2,1,activation1,weights=np.array([[0.6, 0.8]]), bias=0.2)
NN.addLayer(6,50,activation1)
NN.addLayer(50,1,activation1)
#preprocessor.normalize(dataset)
NN.fit(dataset, 500, optimizer,batch_size=124)
exit(1)
#print(NN.FP(x_in=dataset.train[0]))

model = Sequential()
model.add(Dense(50, activation= 'linear' ,input_dim=6))
model.add(Dense(1, activation= 'linear' ))

sgd = SGD(lr=0.005, decay=0, momentum=0.0, nesterov=False)

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

np.random.seed(5)


model.fit(x,y,batch_size=5,epochs=500,shuffle=False)