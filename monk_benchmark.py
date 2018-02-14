
import keras.optimizers as optims
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras import *
from keras.models import Sequential
from NN_lib import preproc
from NN_lib.NN import *
import numpy as np
np.random.seed(5)
outs=1
valr=(0.0,0.0)
clr=0.04
drop=0.0
epochs=1000
inps=17

x_train,y_train = preproc.load_monk("MONK_data/monks-2.train")
x_test,y_test = preproc.load_monk("MONK_data/monks-2.test")

dataset = preproc.Dataset()

dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])


optimizer = Momentum( lr=clr, eps=0.9 ,nesterov=False)

NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=2,activation="tanh", rlambda=valr,regularization="EN",
            dropout=drop,bias=0.0)
NN.addLayer(inputs=2,neurons=outs,activation="tanh",rlambda=valr,regularization="EN",bias=0.0)

#save the weights of our model to load into keras model 
currwg = []
for l in NN.layers:
    w=np.zeros((len(l.W[0])-1,len(l.W)))
    for i in range(0,len(l.W)):
        for j in range(1,len(l.W[i])):
            w[j-1][i] = l.W[i][j]
    currwg.append(w)    #Actual weights
    currwg.append(np.ones(len(l.W))*l.W[i][0]) #Bias


#train our model
(loss, acc, val_loss, val_acc, history)=\
    NN.fit_ds( dataset,epochs, optimizer ,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")


plt.plot(history['tr_loss'], label='loss',ls="-",color="red")

#build and train keras model with our weights
model = Sequential()
model.add(Dense(2  , activation= 'tanh' ,input_dim=inps,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))
model.add(Dropout(drop))
model.add(Dense(outs, activation= 'tanh' ,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))

sgd = optims.SGD(lr=clr, momentum=0.9, decay=0.00,nesterov=False )

model.compile(optimizer=sgd,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

#set keras weights
model.set_weights(currwg)

#fit keras model
his= model.fit(dataset.train[0], dataset.train[1],batch_size=dataset.train[0].shape[0],epochs=epochs,shuffle=True)


plt.plot(his.history['loss'], label='keras loss',ls=":")
plt.title('keras comparison')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':18})
plt.show()