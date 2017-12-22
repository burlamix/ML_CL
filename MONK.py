import preproc
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
import keras.optimizers as optims
from NN import *
from optimizer import *
import validation
import Plotter
import time
from benchmarkMonk import *
import matplotlib.pyplot as plt

#TODO target relativo alla funzione finale di attivazione
#250 160
np.random.seed(11)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = Momentum(lr=0.9,eps=0.9, nesterov=True)

#Check benchmarkMonk file
#bm_monk(optimizer=optimizer3,monk='monk2',act1='tanh',act2='sigmoid',
 #       reg=0.0,bs=169,epochs=1000,trials=1)



x_train,y_train = load_monk("MONK_data/monks-2.train")
x_test,y_test = load_monk("MONK_data/monks-2.test")




dataset = preproc.Dataset()
dataset.init_train([x_train,y_train])
dataset.init_test([x_test,y_test])

#np.random.--------------------seed(525)

optimizer = Adam(lr=0.005,b1=0.9,b2=0.999)
optimizer2 = SimpleOptimizer(lr=0.9)
optimizer3 = Momentum( lr=0.5, eps=0.4 ,nesterov=False)




NN = NeuralNetwork()
NN.addLayer(inputs=17,neurons=2,activation="sigmoid", rlambda=0.0)
NN.addLayer(inputs=2,neurons=1,activation="tanh",rlambda=0.0)
#preprocessor.normalize(dataset)
preproc.Preprocessor().shuffle(dataset)

#dataset.train[0] = np.random.randn(50,17)
#dataset.train[1] = np.random.randn(50,1)

s = time.time()
#(loss, acc, val_loss, val_acc, history)=\
 #   NN.fit_ds( dataset,1000, optimizer2 ,batch_size=169,verbose=3, val_set = dataset.test )

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


#optimizer1 = SimpleOptimizer(lr=0.1)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.5)
optimizer4 = SimpleOptimizer(lr=0.7)


#TODO automatizzare -1 1, 0 -1

acts=[["tanh","tanh"], ["sigmoid","tanh"], ["relu","tanh"]]
opts=[optimizer2,optimizer3,optimizer4]
neurs=[[2,1]]
#print("----senza grid search----",NN.evaluate(x_test,y_test))

fgs = list()
trials = 1
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[700],batch_size=[169], n_layers=2, val_split=0,
                        activations=acts,cvfolds=1,val_set=dataset.test,verbose=2,loss_fun="mse",
                     neurons=neurs ,optimizers=opts)   #with 10 neurons error! i don't now why
    fgs.append(fg)

fgmean = list() #List for holding means

#Create initial configs
for i in fg:
    fgmean.append({'configuration':i['configuration'], 'val_acc':[], 'val_loss':[],
                   'tr_loss':[], 'tr_acc':[]})

#TODO black and white plot(use symbols)
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
#TODO MEDIA SU PIu test vedi k validation
f, (a) = plt.subplots(nrows=len(acts), ncols=len(opts), sharex='col', sharey='row')
i=0

fgforplot=fgmean
hist=False
for row in a:
    for col in row:
        col.set_title('lr:'+str(fgforplot[i]['configuration']['optimizers'].lr)+
                        ',a1:' + fgforplot[i]['configuration']['activations'][0]+
                        ',a2:' + fgforplot[i]['configuration']['activations'][1],fontsize=9)
        if hist:
            col.plot(fgforplot[i]['history']['tr_acc'],label='tr acc',marker='1')
            col.plot(fgforplot[i]['history']['val_acc'],label='val acc')
            col.plot(fgforplot[i]['history']['tr_loss'],label='tr err')
            col.plot(fgforplot[i]['history']['val_loss'],label='val err')
        else:
            col.plot(fgforplot[i]['tr_acc'], label='tr acc',ls=":")
            col.plot(fgforplot[i]['val_acc'], label='val acc',ls="--")
            col.plot(fgforplot[i]['tr_loss'], label='tr err',ls='-')
            col.plot(fgforplot[i]['val_loss'], label='val err')
        col.legend(loc=3,prop={'size':10})
        i+=1
plt.show()
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
model.add(Dense(2, activation= 'tanh' ,kernel_initializer=ini1,input_dim=17,use_bias=True))
model.add(Dense(1, activation= 'tanh',kernel_initializer=ini2 ,use_bias=True))

#input("..")


#sgd = SGD(lr=0.5, decay=0, momentum=0.0, nesterov=False)
sgd = optims.SGD(lr=0.5, momentum=0.0, decay=0.00,nesterov=False )

adada = optims.adam(lr=0.01,beta_1=0.9,beta_2=0.999)

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