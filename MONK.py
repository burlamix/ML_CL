import keras.optimizers as optims
from NN_lib.optimizers import *
from NN_lib.loss_functions import *
from NN_lib import validation
from benchmarkMonk import *
import matplotlib.pyplot as plt
from keras import regularizers
#np.random.seed(11)
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


optimizer = Adam(lr=0.005,b1=0.9,b2=0.999)
optimizer2 = SimpleOptimizer(lr=0.9)
optimizer3 = Momentum( lr=0.9, eps=0.9 ,nesterov=True)
rmsp = RMSProp(lr=0.001,delta=0.9)


#https://elearning.di.unipi.it/pluginfile.php/15587/mod_resource/content/2/ML-17-NN-part2-v0.23.pdfx
#TODO LR decay (more important with mini batch)
NN = NeuralNetwork()
NN.addLayer(inputs=17,neurons=2,activation="tanh", rlambda=0.0001,regularization="L1",bias=0)
NN.addLayer(inputs=2,neurons=1,activation="sigmoid",rlambda=0.0001,regularization="L1",bias=0)

dataset.train[0] = np.random.randn(50,17)
dataset.train[1] = np.random.randn(50,1)

currwg = []
for l in NN.layers:
    w=np.zeros((len(l.W[0])-1,len(l.W)))
    for i in range(0,len(l.W)):
        for j in range(1,len(l.W[i])):
            w[j-1][i] = l.W[i][j]
    currwg.append(w)    #Actual weights
    currwg.append(np.zeros(len(l.W))) #Bias

(loss, acc, val_loss, val_acc, history)=\
    NN.fit_ds( dataset,1000, optimizer2 ,batch_size=169,verbose=0, val_set = dataset.test,loss_func="mae")


print(history['val_acc'])
#history['val_loss']
#history['tr_acc']
#history['tr_loss']
#plt.plot(history['val_acc'])
#plt.plot(history['val_loss'])
plt.plot(history['tr_acc'], label='tr_acc',ls="-",)
plt.plot(history['tr_loss'], label='tr_loss',ls="-",color="red")

#plt.show()



ini1 = keras.initializers.RandomUniform(minval=-0.7 / 17, maxval=0.7 / 17, seed=None)
ini2 = keras.initializers.RandomUniform(minval=-0.7 / 2, maxval=0.7 / 2, seed=None)

model = Sequential()
model.add(Dense(2, activation= 'tanh' ,kernel_initializer=ini1,input_dim=17,use_bias=True,bias_initializer="zeros",kernel_regularizer=regularizers.l1(0.001),bias_regularizer=regularizers.l1(0.001)))
model.add(Dense(1, activation= 'sigmoid',kernel_initializer=ini2 ,use_bias=True,bias_initializer="zeros",kernel_regularizer=regularizers.l1(0.001),bias_regularizer=regularizers.l1(0.001)))


sgd = optims.SGD(lr=0.9, momentum=0.0, decay=0.00,nesterov=False )
adada = optims.adam(lr=0.005,beta_1=0.9,beta_2=0.999)
rmsp = optims.RMSprop(lr=0.001,rho=0.9,epsilon=1e-6)

model.compile(optimizer=sgd,
              loss= 'mean_absolute_error' ,
              metrics=[ 'accuracy' ])

model.set_weights(currwg)

s=0
#for l in model.layers:
#    s+=np.sum(np.abs(l.get_weights()))
#print(np.sum(s))


his= model.fit(dataset.train[0], dataset.train[1],batch_size=169,epochs=1000,shuffle=True)
#print(model.evaluate(x_test, y_test))
print(his.history.keys())
plt.plot(his.history['acc'], label='tr_loss_k',ls=":")
plt.plot(his.history['loss'], label='tr_loss_k',ls=":")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()
###################################################################### sopra a qui è per confrontare keras 

print(mse(NN.predict(dataset.train[0]),dataset.train[1]))
print(NN.evaluate(dataset.train[0],dataset.train[1]))
exit(1)
s=0
for l in NN.layers:
    s+=np.sum(np.abs(l.W))
print(np.sum(s))

#Plotter.loss_over_epochs(history)


#optimizer1 = SimpleOptimizer(lr=0.1)
optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.5)
optimizer4 = SimpleOptimizer(lr=0.7)
optimizer5 = Momentum(lr=0.5,eps=0.5)
optimizer6 = Momentum(lr=0.5,eps=0.9)
optimizer7 = Momentum(lr=0.3,eps=0.5)
optimizer8 = Momentum(lr=0.3,eps=0.9)
optimizer9 = Adam(lr=0.005,b1=0.9,b2=0.999)
optimizer10 = RMSProp(lr=0.0051)


acts=[["tanh","tanh"], ["sigmoid","tanh"]]
opts=[optimizer9,optimizer10]#,optimizer6,optimizer7,optimizer8]
neurs=[[2,1]]
#print("----senza grid search----",NN.evaluate(x_test,y_test))

fgs = list()
trials = 1
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[700], batch_size=[169], n_layers=2, val_split=0,
                                               activations=acts, cvfolds=1, val_set=dataset.test, verbose=2, loss_fun="mse",
                                               neurons=neurs, optimizers=opts)   #with 10 neurons error! i don't now why
    fgs.append(fg)
exit(1)

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
            col.plot(fgforplot[i]['tr_loss'], label='tr err',ls='-.')
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




ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/20), seed=None)
ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2/4), seed=None)


model = Sequential()
model.add(Dense(2, activation= 'tanh' ,kernel_initializer=ini1,input_dim=17,use_bias=True))
model.add(Dense(1, activation= 'tanh',kernel_initializer=ini2 ,use_bias=True))

#input("..")

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