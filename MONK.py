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
import validation

x_train,y_train = load_monk("MONK_data/monks-3.train")
x_test,y_test = load_monk("MONK_data/monks-3.test")


optimizer = SimpleOptimizer(lr=0.9)
activation = Activation(sigmoid,sigmoddxf)
activation1 = Activation(linear,lineardxf)
dataset = preproc.Dataset()
dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])

NN = NeuralNetwork()
#NN.addLayer(2,1,activation1,weights=np.array([[0.6, 0.8]]), bias=0.2)
NN.addLayer(6,100,activation)
NN.addLayer(100,1,activation)
#preprocessor.normalize(dataset)

NN.fit_ds( dataset, 500, optimizer,batch_size=124,verbose=0)

print("----senza grid search----",NN.evaluate(x_test,y_test))

fg,grid_res, pred = validation.grid_search(dataset, epochs=[500],batch_size=[124], n_layers=2, val_split=0,activations=[[activation]*2],
                       neurons=[[50,1],[100,1],[150,1]] ,optimizers=[optimizer])   #with 10 neurons error! i don't now why

print("-------grid search-------",grid_res.NN.evaluate(x_test,y_test))



#TODO  asseconda dell'ordine con cui metti i valori di un parametro per fare il grid search cambia il risultati finale...!!!!!!!!!!!!!!





y_hot = keras.utils.to_categorical(y_train, num_classes=2)


model = Sequential()
model.add(Dense(11, activation= 'sigmoid' ,input_dim=6))
model.add(Dense(2, activation= 'sigmoid' ))

sgd = SGD(lr=0.1, decay=0, momentum=0.0, nesterov=False)

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

#np.random.seed(5)



#model.fit(x,y_hot,batch_size=124,epochs=500,shuffle=False)