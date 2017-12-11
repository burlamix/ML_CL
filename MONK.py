import preproc
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
from NN import *
from optimizer import *
import validation

x_train,y_train = load_monk("MONK_data/monks-1.train")
x_test,y_test = load_monk("MONK_data/monks-1.test")


optimizer = SimpleOptimizer(lr=0.5)
activation = Activation(sigmoid,sigmoddxf)
activation1 = Activation(linear,lineardxf)
dataset = preproc.Dataset()
dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])

NN = NeuralNetwork()
#NN.addLayer(2,1,activation1,weights=np.array([[0.6, 0.8]]), bias=0.2)
NN.addLayer(17,3,activation)
#NN.addLayer(60,60,activation)
NN.addLayer(3,1,activation)
#preprocessor.normalize(dataset)

print(dataset.train[0].shape)
print(dataset.train[1].shape)
(tloss,vloss)=NN.fit_ds( dataset, 500, optimizer,batch_size=32,verbose=2, val_split=25)
print("----senza grid search----",tloss,vloss)#NN.evaluate(x_test,y_test))
#fg,grid_res, pred = validation.grid_search(dataset, epochs=[500],batch_size=[124], n_layers=2, val_split=0,
#                        activations=[[activation]*2, [activation1,activation]],
#                       neurons=[[100,1],[50,1],[150,1]] ,optimizers=[optimizer])   #with 10 neurons error! i don't now why

print("-------grid search-------")#grid_res.NN.evaluate(x_test,y_test))
#for i in fg:
#    print(i['val_loss'])



#y_hot = keras.utils.to_categorical(y_train, num_classes=2)


model = Sequential()
model.add(Dense(3, activation= 'sigmoid' ,kernel_initializer='normal',input_dim=17))
model.add(Dense(1, activation= 'sigmoid',kernel_initializer='normal' ))

#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = SGD(lr=0.5, momentum=0.0 )

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

#np.random.seed(5)


model.fit(x_train,y_train, validation_split=0.25,batch_size=124,epochs=5000,shuffle=False)
