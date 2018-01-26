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
valr =(0,0)
clr=0.01
drop=0.0
epochs=100
inps=10
#13770039132
np.random.seed(23185)
dataset = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"


dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))

optimizer = Adine( lr=clr, ls=None)
adam1 = Adam(lr=clr,b1=0.9,b2=0.999)
adamax1 = Adamax(lr=clr,b1=0.9,b2=0.999)
m1 = Momentum(lr=clr,eps=0.9,nesterov=True)
optimizer = SimpleOptimizer( lr=clr, ls=None)
optimizer = Adam(lr=clr,b1=0.9,b2=0.999)
optimizer = RMSProp(lr=clr,delta= 0.1)


NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=25,activation="tanh", rlambda=valr,regularization="EN",
            dropout=0,bias=0.0)
NN.addLayer(inputs=25,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)

#save the weights of our model to load into keras model 
currwg = []
for l in NN.layers:
    w=np.zeros((len(l.W[0])-1,len(l.W)))
    for i in range(0,len(l.W)):
        for j in range(1,len(l.W[i])):
            w[j-1][i] = l.W[i][j]
    currwg.append(w)    #Actual weights
    currwg.append(np.ones(len(l.W))*l.W[i][0]) #Bias

(loss, acc, val_loss, val_acc, history2)=\
    NN.fit_ds( dataset,epochs, optimizer  ,val_split=0,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

model = Sequential()
model.add(Dense(25, activation= 'tanh' ,use_bias=True,input_dim=10,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))
model.add(Dense(outs, activation= 'linear' ,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))

sgd = optims.SGD(lr=clr, momentum=0, decay=0.00,nesterov=False )
sgd = optims.Adam(lr=clr,epsilon=1e-8)
sgd = optims.RMSprop(lr=clr, rho=0.9,epsilon=1e-6, decay=0.00 )


model.compile(optimizer=sgd,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

model.set_weights(currwg)
history = model.fit(dataset.train[0], dataset.train[1],batch_size=dataset.train[0].shape[0],epochs=epochs,shuffle=True)

print(history.history.keys())
plt.plot(history.history['loss'], label='loss',ls="-",color="red")
plt.plot(history2['tr_loss'], label='loss',ls="-",color="blue")
#plt.axes().set_ylim([0,20])

plt.show()

