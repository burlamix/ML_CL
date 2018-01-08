import keras.optimizers as optims
from NN_lib.optimizers import *
from NN_lib.loss_functions import *
from NN_lib import validation
from benchmarkMonk import *
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
#np.random.seed(1)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = Momentum(lr=0.5,eps=0.9, nesterov=True)
#Check benchmarkMonk file

outs=1
valr2=(0.0,0.0)
valr=0.0
clr=0.04
drop=0.0
drop2=0.0
epochs=1000
inps=17

x_train,y_train = load_monk("MONK_data/monks-2.train")
x_test,y_test = load_monk("MONK_data/monks-2.test")
train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"

dataset = preproc.Dataset()

dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])


########################################
optimizer = Adam(lr=clr,b1=0.9,b2=0.999)
optimizer2 = SimpleOptimizer(lr=clr)
optimizer3 = Momentum( lr=clr, eps=0.9 ,nesterov=False)
rmsp = RMSProp(lr=clr,delta=0.9)



NN = NeuralNetwork()
NN.addLayer(inputs=inps,neurons=2,activation="tanh", rlambda=valr2,regularization="EN",
            dropout=drop,bias=0.0)
NN.addLayer(inputs=2,neurons=outs,activation="tanh",rlambda=valr2,regularization="EN",
            dropout=drop2,bias=0.0)


#save layer weight for comparison with keras 
currwg = []
for l in NN.layers:
    w=np.zeros((len(l.W[0])-1,len(l.W)))
    for i in range(0,len(l.W)):
        for j in range(1,len(l.W[i])):
            w[j-1][i] = l.W[i][j]
    currwg.append(w)    #Actual weights
    currwg.append(np.ones(len(l.W))*l.W[i][0]) #Bias

(loss, acc, val_loss, val_acc, history)=\
    NN.fit_ds( dataset,epochs, optimizer3 ,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

print('init',np.sum([np.sum(np.abs(a)) for a in NN.get_weight()]))

#history['val_loss']
#history['tr_acc']
#history['tr_loss']
#plt.plot(history['val_acc'])
#plt.plot(history['val_loss'])
#plt.plot(history['tr_acc'], label='tr_acc',ls="-",)
plt.plot(history['tr_loss'], label='loss',ls="-",color="red")

#plt.show()

ini1 = keras.initializers.RandomUniform(minval=-0.7 / 17, maxval=0.7 / 17, seed=None)
ini2 = keras.initializers.RandomUniform(minval=-0.7 / 2, maxval=0.7 / 2, seed=None)

model = Sequential()
model.add(Dense(2  , activation= 'tanh' ,kernel_initializer=ini1,input_dim=inps,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr2[0],valr2[1]),bias_regularizer=regularizers.l1_l2(valr2[0],valr2[1])))
model.add(Dropout(drop))
model.add(Dense(outs, activation= 'tanh',kernel_initializer=ini2 ,use_bias=True,
    bias_initializer="zeros",kernel_regularizer=regularizers.l1_l2(valr2[0],valr2[1]),bias_regularizer=regularizers.l1_l2(valr2[0],valr2[1])))

sgd = optims.SGD(lr=clr, momentum=0.9, decay=0.00,nesterov=False )
adada = optims.adam(lr=clr,beta_1=0.9,beta_2=0.999)
rmsp = optims.RMSprop(lr=clr,rho=0.9,epsilon=1e-6)

model.compile(optimizer=sgd,
              loss= 'mean_squared_error' ,
              metrics=[ 'accuracy' ])

model.set_weights(currwg)

s=0


#fit keras model
his= model.fit(dataset.train[0], dataset.train[1],batch_size=dataset.train[0].shape[0],epochs=epochs,shuffle=True)

print('init',np.sum([np.sum(np.abs(a)) for a in model.get_weights()]))

print(his.history.keys())
#plt.plot(his.history['acc'], label='tr_loss_k',ls=":")
plt.plot(his.history['loss'], label='keras loss',ls=":")
plt.title('keras comparison')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right',prop={'size':18})
plt.show()

exit(1)
#data analysis

print(mse(NN.predict(dataset.train[0]),dataset.train[1]))
print(NN.evaluate(dataset.train[0],dataset.train[1]))

exit(1)

s=0
for l in NN.layers:
    s+=np.sum(np.abs(l.W))
print(np.sum(s))


#grid search
optimizer1 = SimpleOptimizer(lr=0.1)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.2)
optimizer4 = SimpleOptimizer(lr=0.7)
optimizer41 = SimpleOptimizer(lr=0.9)
optimizer5 = Momentum(lr=0.1,eps=0.5,nesterov=True)
optimizer6 = Momentum(lr=0.2,eps=0.5,nesterov=True)
optimizer7 = Momentum(lr=0.3,eps=0.5)
optimizer8 = Momentum(lr=0.3,eps=0.9)
optimizer9 = Adam(lr=0.005,b1=0.9,b2=0.999)
optimizer10 = RMSProp(lr=0.0051)


acts=[["sigmoid","tanh"],["tanh","tanh"]]
opts=[optimizer5,optimizer6]
neurs=[[2,1]]
rlambda=[[0.0,0.0]]

fgs = list()
trials = 10
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[2000], batch_size=[169], n_layers=2, val_split=0,
                                               activations=acts, cvfolds=1, val_set=dataset.test, verbose=1, loss_fun="mse",val_loss_fun="mse",rlambda=rlambda,
                                               neurons=neurs, optimizers=opts)   #with 10 neurons error! i don't now why
    fgs.append(fg)

fgmean = list() #List for holding means

#Create initial configs
for i in fg:
    fgmean.append({'configuration':i['configuration'], 'val_acc':[], 'val_loss':[],
                   'tr_loss':[], 'tr_acc':[]})

#Sum up the contributions from each trial
for fullgrid in fgs:
    for i in fullgrid:
        for j in range(0,len(fgmean)):
            if i['configuration']==fgmean[j]['configuration']:
                if fgmean[j]['val_acc']!=[]:
                    fgmean[j]['val_acc']+=np.array(i['history']['val_acc'])
                    fgmean[j]['val_loss']+=np.array(i['history']['val_loss'])
                    fgmean[j]['tr_acc']+=np.array(i['history']['tr_acc'])
                    fgmean[j]['tr_loss']+=np.array(i['history']['tr_loss'])
                else:
                    fgmean[j]['val_acc']=np.array(i['history']['val_acc'])
                    fgmean[j]['val_loss']=np.array(i['history']['val_loss'])
                    fgmean[j]['tr_acc']=np.array(i['history']['tr_acc'])
                    fgmean[j]['tr_loss']=np.array(i['history']['tr_loss'])
                break

for i in range(0,len(fgmean)):
    fgmean[i]['val_acc']/=trials
    fgmean[i]['val_loss']/=trials
    fgmean[i]['tr_acc']/=trials
    fgmean[i]['tr_loss']/=trials


nconfig = len(acts)*len(opts)*len(neurs)

f, (a) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row',squeeze=False)
i=0

fgforplot=fgmean
hist=False
for row in a:
    for col in row:
        col.set_title('lr:'+str(fgforplot[i]['configuration']['optimizers'].lr)+
                        ',a1:' + fgforplot[i]['configuration']['activations'][0]+
                        ',a2:' + fgforplot[i]['configuration']['activations'][1]+
                        ',batch_size:' + str(fgforplot[i]['configuration']['batch_size']),fontsize=9)
        if hist:
            col.plot(fgforplot[i]['history']['tr_acc'],label='tr acc',marker='1')
            col.plot(fgforplot[i]['history']['val_acc'],label='val acc')
            col.plot(fgforplot[i]['history']['tr_loss'],label='tr err')
            col.plot(fgforplot[i]['history']['val_loss'],label='val err')
        else:
            col.plot(fgforplot[i]['tr_acc'], label='tr acc',ls=":")
            col.plot(fgforplot[i]['val_acc'], label='val acc',ls="--")
            col.plot(fgforplot[i]['tr_loss'], label='tr err',ls='-.')
            col.plot(fgforplot[i]['val_loss'], label='val err')
        col.legend(loc="best",prop={'size':10})
        i+=1
plt.show()

