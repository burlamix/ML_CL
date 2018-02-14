import numpy as np
from NN_lib.optimizers import *
from NN_lib.NN import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers as optims

outs=2 #number of outputs
valr =(0,0) #regularization (l1,l2)
clr=0.01 #step size
drop=0.0 #dropout
neurons=25 #number of neurons in the hidden layer
epochs=250
inps=10 #number of input features

#Generate a random dataset
x = np.random.randn(500,inps)
y = np.random.randn(500,outs)


#Some sample keras optimizers
optimizer_keras = optims.SGD(lr=clr, momentum=0.9, decay=0.00,nesterov=False )
optimizer_keras = optims.RMSprop(lr=clr, rho=0.9,epsilon=1e-6, decay=0.00 )
optimizer_keras = optims.Adamax(lr=clr,epsilon=1e-8)

#Some sample optimizers
optimizer = Adam(lr=clr,b1=0.9,b2=0.999)
optimizer = Momentum(lr=clr,eps=0.9,nesterov=False)
optimizer = RMSProp(lr=clr,delta= 0.9)
optimizer = Adamax(lr=clr,b1=0.9,b2=0.999)

#Create neural network
NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=neurons,activation="tanh", rlambda=valr,regularization="EN",
            dropout=0.0,bias=0.0)
NN.addLayer(inputs=neurons,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)

#Save the weights of our model to load into keras model
currwg = []
for l in NN.layers:
    w=np.zeros((len(l.W[0])-1,len(l.W)))
    for i in range(0,len(l.W)):
        for j in range(1,len(l.W[i])):
            w[j-1][i] = l.W[i][j]
    currwg.append(w)    #Actual weights
    currwg.append(np.ones(len(l.W))*l.W[i][0]) #Bias

#Train our model
(loss, acc, val_loss, val_acc, history2)=\
    NN.fit( x,y,epochs, optimizer  ,val_split=0,batch_size=x.shape[0],verbose=2,loss_func="mse")

#Create keras' model
model = Sequential()
model.add(Dense(neurons, activation= 'tanh' ,use_bias=True,input_dim=10,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))
model.add(Dense(outs, activation= 'linear' ,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))

model.compile(optimizer=optimizer_keras,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

#Load the initial weights of our model (i.e. before training) into keras
model.set_weights(currwg)

#Train keras;
history = model.fit(x, y,batch_size=x.shape[0],epochs=epochs,shuffle=True)

print(history.history.keys())
plt.plot(history.history['loss'], label='keras',ls="--",color="red")
plt.plot(history2['tr_loss'], label='us',ls="-.",color="blue")
#plt.axes().set_ylim([0,20])
plt.title(optimizer.pprint())
plt.legend(loc='upper right',prop={'size':13})
plt.show()

