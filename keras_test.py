import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD




model = Sequential()
model.add(Dense(20, activation= 'linear' ,input_dim=10))
model.add(Dense(1, activation= 'linear' ))

sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

np.random.seed(5)


x=np.random.rand(500,10)
y=np.random.rand(500,1)
model.fit(x,y,batch_size=500,epochs=10000,shuffle=True)