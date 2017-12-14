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
import Plotter

#TODO target relativlo alla funzione finale di attivazione
#usare 0.9 per la saturazione !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#plottare MSE  per il cuore del modello

np.random.seed(5)

x_train,y_train = load_monk("MONK_data/monks-1.train")
x_test,y_test = load_monk("MONK_data/monks-1.test")


optimizer = SimpleOptimizer(lr=0.5)
#momentum dei tipo 0.9
dataset = preproc.Dataset()
dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])

NN = NeuralNetwork()
#NN.addLayer(2,1,activation1,weights=np.array([[0.6, 0.8]]), bias=0.2)
NN.addLayer(17,3,"tanh", rlambda=0.0)
#NN.addLayer(60,60,activation)
NN.addLayer(3,1,"tanh",rlambda=0.0)
#preprocessor.normalize(dataset)

print(dataset.train[0].shape)
print(dataset.train[1].shape)

(loss, acc, val_loss, val_acc, history)=NN.fit_ds( dataset, 1000, optimizer,batch_size=32,verbose=3)

#check weight 
s=0
for l in NN.layers:
    s+=np.sum(np.abs(l.W))
print(np.sum(s))

#print(NN.FP(x_in=dataset.train[0]))

#Plotter.loss_over_epochs(history)



#print("----senza grid search----",NN.evaluate(x_test,y_test))
#fg,grid_res, pred = validation.grid_search(dataset, epochs=[500],batch_size=[124], n_layers=2, val_split=0,
#                        activations=[[activation]*2, [activation1,activation]],
#                       neurons=[[100,1],[50,1],[150,1]] ,optimizers=[optimizer])   #with 10 neurons error! i don't now why

#print("-------grid search-------")#grid_res.NN.evaluate(x_test,y_test))
#for i in fg:
#    print(i['val_loss'])



#y_hot = keras.utils.to_categorical(y_train, num_classes=2)
ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/20), seed=None)
ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/4), seed=None)


model = Sequential()
model.add(Dense(2, activation= 'tanh' ,kernel_initializer=ini1,input_dim=17,use_bias=False))
model.add(Dense(1, activation= 'tanh',kernel_initializer=ini2 ,use_bias=False))




#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = SGD(lr=0.5, momentum=0.0, decay=0.00 )

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

s=0
for l in model.layers:
    s+=np.sum(np.abs(l.get_weights()))
#àprint(np.sum(s))
#np.random.seed(5)


#model.fit(x_train, y_train,batch_size=32,epochs=10000,shuffle=False)
#print(model.evaluate(x_test, y_test))