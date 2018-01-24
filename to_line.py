import numpy as np
from NN_lib.optimizers import *
from NN_lib import loss_functions
from NN_lib.NN import *
import matplotlib.pyplot as plt
from NN_lib import preproc
from NN_lib import linesearches
import lss
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers as optims
outs=2
valr=(0,0)
clr=0.001
drop=0.0
epochs=500
inps=10
#13770039132
np.random.seed(23185)
dataset = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"


dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))
#dataset.train[0]= np.random.randn(11,10)
#dataset.train[1]= np.random.randn(11,2)
#j = dataset.train[0]
#j = np.concatenate((np.ones((j.shape[0],1)),j),axis=1)

#print(np.linalg.cond(np.eye((10))+np.dot(dataset.train[0].T,dataset.train[0])))
amg = linesearches.armj_wolfe(m1=1e-4, m2=0.9, lr=clr, min_lr=1e-11, scale_r=0.9, max_iter=100)
#amg = linesearches.back_track(lr=1, m1=1e-4, scale_r=0.1, min_lr=1e-11, max_iter=100)

#optimizer = ConjugateGradient( lr=clr, ls=amg)
optimizer = SimpleOptimizer( lr=clr, ls=None)
optimizer = Adine( lr=clr, ls=None)
#optimizer = Momentum( lr=clr)
#optimizer = Adam(lr=0.01 )

#Try LLSQ on the cup dataset

#NN = NeuralNetwork()
#NN.addLayer(inputs=inps,neurons=15,activation="tanh", rlambda=valr,regularization="EN",
#            dropout=0.0,bias=0.0)
#NN.addLayer(inputs=15,dropout=0,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)
#weights = NN.get_weights()
#train our model
#(loss, acc, val_loss, val_acc, history)=\
#    NN.fit_ds( dataset,epochs, optimizer ,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=25,activation="tanh", rlambda=valr,regularization="EN",
            dropout=0,bias=0.0)
NN.addLayer(inputs=25,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)
#NN.set_weights(weights)
(loss, acc, val_loss, val_acc, history2)=\
    NN.fit_ds( dataset,epochs, optimizer  ,val_split=0,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

model = Sequential()
model.add(Dense(outs, activation= 'linear' ,use_bias=True,input_dim=10,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))
sgd = optims.SGD(lr=clr, momentum=0.9, decay=0.00,nesterov=False )
model.compile(optimizer=sgd,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])
#model.fit(dataset.train[0], dataset.train[1],batch_size=dataset.train[0].shape[0],epochs=1000,shuffle=True)


w = NN.get_weights()
j = dataset.train[0]
j = np.concatenate((np.ones((j.shape[0],1)),j),axis=1)

#PRED WITH NUMPY SOLVER
sol = np.linalg.lstsq(j,dataset.train[1])
pred = np.dot(j,sol[0])

#PRED WITH OUR SOLVER
xsolv = lss.LLSQ1R(j,dataset.train[1])
predus = np.dot(j,xsolv)
xsolvqr = lss.LLSQ(j,dataset.train[1])
predusqr = np.dot(j,xsolvqr)

predusnet = np.dot(j,NN.get_weights()[0])

nnres = loss_functions.mse(dataset.train[1],NN.predict(dataset.train[0]))
npres = loss_functions.mse(dataset.train[1],pred)
usres = loss_functions.mse(dataset.train[1],predus)
usresqr = loss_functions.mse(dataset.train[1],predusqr)
usnetres = loss_functions.mse(dataset.train[1],predusnet)

print('nn',nnres)
print('numpy',npres)
print('us',usres)
print('usqr',usresqr)
print('usnet',usnetres)
print('diff',npres-usres)
print('diff',nnres-usres)
print(NN.get_weights())
print('------------------')
print(xsolv)
#plt.plot(history['tr_loss'], label='loss',ls="-",color="red")
plt.plot(history2['tr_loss'], label='loss',ls="-",color="blue")
plt.plot(history2['val_loss'], label='vloss',ls="-",color="red")
plt.axes().set_ylim([0,20])

plt.show()
'''
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
'''