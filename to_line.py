import sys
import os
sys.path.append(os.getcwd()+"/NN_lib")
import ptimizers 
import loss_functions 
import NN 
import matplotlib.pyplot as plt
import preproc

outs=2
valr=(0.00,0.000)
clr=0.01
drop=0.0
epochs=1000
inps=10

np.random.seed(5)
dataset = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"


dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))



optimizer = SimpleOptimizer( lr=clr)
optimizer2 = LineSearchOptimizer( lr=clr)


NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=15,activation="tanh", rlambda=valr,regularization="EN",
            dropout=0.0,bias=0.0)
NN.addLayer(inputs=15,dropout=0,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)
weights = NN.get_weights()
#train our model
#(loss, acc, val_loss, val_acc, history)=\
#    NN.fit_ds( dataset,epochs, optimizer ,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=50,activation="sigmoid", rlambda=valr,regularization="EN",
            dropout=0,bias=0.0)
NN.addLayer(inputs=15,neurons=outs,activation="linear",rlambda=valr,regularization="EN",bias=0.0)
NN.set_weights(weights)
(loss, acc, val_loss, val_acc, history2)=\
    NN.fit_ds( dataset,epochs, optimizer2 ,val_split=30,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")


#plt.plot(history['tr_loss'], label='loss',ls="-",color="red")
plt.plot(history2['tr_loss'], label='loss',ls="-",color="blue")
plt.plot(history2['val_loss'], label='vloss',ls="-",color="red")
plt.axes().set_ylim([0,2])

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