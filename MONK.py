import preproc
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
import keras.optimizers as opts
from NN import *
from optimizer import *
import validation
import Plotter
import time
from benchmarkMonk import *
#TODO target relativo alla funzione finale di attivazione



#plottare MSE  per il cuore del modello
optimizer2 = SimpleOptimizer(lr=0.5)

#Check benchmarkMonk file
bm_monk(optimizer=optimizer2,monk='monk1',act1='tanh',act2='sigmoid',
        reg=0.0,bs=8,epochs=1000,trials=30)
exit(1)

x_train,y_train = load_monk("MONK_data/monks-1.train")
x_test,y_test = load_monk("MONK_data/monks-1.test")

dataset = preproc.Dataset()
dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])

#np.random.seed(525)

optimizer = Adam(lr=0.001,b1=0.9,b2=0.999)
optimizer2 = SimpleOptimizer(lr=0.5)
optimizer3 = Momentum( lr=0.3, eps=0.9 ,nesterov=True)




NN = NeuralNetwork()
NN.addLayer(17,3,"tanh", rlambda=0.0)
NN.addLayer(3,1,"sigmoid",rlambda=0.0)
#preprocessor.normalize(dataset)
#preproc.Preprocessor().shuffle(dataset)

#dataset.train[0] = np.random.randn(50,17)
#dataset.train[1] = np.random.randn(50,1)

s = time.time()
#(loss, acc, val_loss, val_acc, history)=\
 #   NN.fit_ds( dataset, 1500, optimizer2 ,batch_size=32,verbose=3, val_set = dataset.test )

t=time.time()-s
print('time:',t)






#input('..')
#check weight



s=0
for l in NN.layers:
    s+=np.sum(np.abs(l.W))
print(np.sum(s))

#print(NN.FP(x_in=dataset.train[0]))

#Plotter.loss_over_epochs(history)


optimizer1 = SimpleOptimizer(lr=0.1)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.5)
optimizer4 = SimpleOptimizer(lr=0.7)


#TODO automatizzare -1 1, 0 -1

#print("----senza grid search----",NN.evaluate(x_test,y_test))
#fg,grid_res, pred = validation.grid_search(dataset, epochs=[10000],batch_size=[124], n_layers=2, val_split=0,
 #                       activations=[["tanh","sigmoid"], ["sigmoid","sigmoid"]],
  #                     neurons=[[100,1],[9,1],[20,1],[3,1],[50,1]] ,optimizers=[optimizer1,optimizer2,optimizer3,optimizer4])   #with 10 neurons error! i don't now why

#print("-----------------------")
#for i in fg:
    #print(i["configuration"])
    #print(i["val_loss"])
    #print("storia e val acc ",i["history"]["val_loss"])
    #print("++++++\n")
 #   None

#print(grid_res.neurons)
#print(grid_res.activations)









#y_hot = keras.utils.to_categorical(y_train, num_classes=2)
ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/20), seed=None)
ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/4), seed=None)


model = Sequential()
model.add(Dense(3, activation= 'tanh' ,kernel_initializer=ini1,input_dim=17,use_bias=True))
model.add(Dense(1, activation= 'sigmoid',kernel_initializer=ini2 ,use_bias=True))

#input("..")


#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = opts.SGD(lr=0.5, momentum=0.0, decay=0.00,nesterov=False )

adada = opts.adam(lr=0.01,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=sgd,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

s=0
#for l in model.layers:
#    s+=np.sum(np.abs(l.get_weights()))
#print(np.sum(s))


model.fit(dataset.train[0], dataset.train[1],batch_size=124,epochs=500,shuffle=False)
print(model.evaluate(x_test, y_test))


#2 functions or param (want grad or not)

print(" \n\n done ")