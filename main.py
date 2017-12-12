import sys
from sklearn import preprocessing
from preproc import *
from NN import *
from optimizer import *
from validation import *
train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
np.random.seed(5)
dataset = Dataset()
dataset.init_train(load_data(train_data_path, True, header_l=10, targets=2))
dataset.init_test(load_data(test_data_path, False, header_l=10))

preprocessor = Preprocessor()
preprocessor.normalize(dataset,norm_output=False)


#preprocessing




        #gradient calculation is the same for all optimizers,
        #the update rule changes for each one
        #while backprop set layer.gradient
     #   self.optimizer.optimize(self.loss_func(dataset.train))


#Real dataset
# test with only string-------------------------------------------------------------------






#NN.fit_ds(dataset, 100, optimizer, batch_size=1016)
#dataset.train[0] = np.random.rand(5000,10)
#dataset.train[1] = np.random.rand(5000,2)



model = Sequential()
model.add(Dense(55, activation= 'linear' ,input_dim=10))
model.add(Dense(2, activation= 'linear' ))

sgd = SGD(lr=0.0005, decay=0, momentum=0.0, nesterov=False)

model.compile(optimizer= sgd ,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

np.random.seed(5)



#model.fit(dataset.train[0],dataset.train[1],batch_size=32,epochs=500,shuffle=False)

optimizer = SimpleOptimizer(0.0005)

NN = NeuralNetwork()#276828657
NN.addLayer(10,55,"linear")
NN.addLayer(55,2,"linear")

NN.fit_ds(dataset, epochs=100, val_split=30, verbose=2, optimizer=optimizer)

#a = NN.evaluate(dataset)
#fg,grid_res, pred = grid_search(dataset, epochs=[100], n_layers=2, val_split=30,
#                       neurons=[[2,2],[55,2],[20,2],[25,2],[30,2]] ,optimizers=[optimizer],rlambda=[[0.1,0.1]])   #with 10 neurons error! i don't now why

#grid_res.fit(dataset,100,1016)
#print(grid_res.neurons)
#b = grid_res.NN.evaluate(dataset)
#print(a)#result normal fit
#print(b)#result from grid search
#print("tloss:"+str(grid_res.NN.evaluate(dataset.train[0],dataset.train[1])))
#for i in fg:
#    print(i['val_loss'])

#print("PRED:"+str(len(pred)))
#NN.fit_ds(dataset, 100, optimizer=optimizer, val_split=0,verbose=2)





#check weight 
#s=0
#for l in NN.layers:
#    s+=np.sum(np.abs(l.W))
#print(np.sum(s))
#print(NN.FP(x_in=dataset.train[0]))





#TODO min max when min=max -> if min=max then normalize as el/(length of list)
#TODO make dataset work when input is array and not matrix
#TODO gradient checking http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
#TODO compare with keras setting seed
