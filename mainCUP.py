import preproc
from NN_lib import validation
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from NN_lib import regularizations
np.random.seed(915)
dataset = preproc.Dataset()
dataset2016 = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"
train_2016 = "data/LOC-OSM2-TR.csv"
test_2016 = "data/LOC-OSM2-TS.csv"

dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))

dataset2016.init_train(preproc.load_data(train_2016, True, header_l=10, targets=2))
dataset2016.init_test(preproc.load_data(test_2016, False, header_l=10))

optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.02)
optimizer4 = SimpleOptimizer(lr=0.7)
optimizer5 = Momentum(lr=0.5,eps=0.5)
optimizer6 = Momentum(lr=0.5,eps=0.9)
optimizer7 = Momentum(lr=0.3,eps=0.5)
optimizer8 = Momentum(lr=0.3,eps=0.9)
optimizer9 = Adam(lr=0.05,b1=0.9,b2=0.999)
optimizer10 = RMSProp(lr=0.0151)

preprocessor = preproc.Preprocessor()
#preprocessor.remove_outliers(dataset)
#print(preprocessor.remove_outliers(dataset2016))
preprocessor.shuffle(dataset)
#preprocessor.normalize(dataset,method='minmax')
acts=[["relu","linear"],["tanh","linear"],["sigmoid","linear"]]
opts=[optimizer9]#,optimizer6,optimizer7,optimizer8]
neurs=[[5,2],[20,2],[80,2]]
batches = [dataset.train[0].shape[0]]
losses = ["mse"]
regs = [[regularizations.reguls["EN"],regularizations.reguls["EN"]]]
rlambdas = [[(0,0),(0,0)]]

fgs = list()
trials = 1
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[5], batch_size=batches,
                                               n_layers=2, val_split=20,activations=acts,
                                               regularizations=regs, rlambda=rlambdas,
                                               cvfolds=1, val_set=None, verbose=2,
                                               loss_fun=losses, val_loss_fun="mse",
                                               neurons=neurs, optimizers=opts)
    fgs.append(fg)
#print(grid_res.NN.evaluate(dataset2016.train[0],dataset2016.train[1],"mee"))
#exit(1)
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
f, (a) = plt.subplots(nrows=len(acts)*len(batches)*len(neurs), ncols=len(opts)*len(losses), sharex='col', sharey='row',squeeze=False)
i=0

fgforplot=fgmean
hist=False
for row in a:
    for col in row:
        col.set_title('tloss:'+str(fgforplot[i]['configuration']['loss_fun'])+
                      ',vloss:'+str(fgforplot[i]['configuration']['val_loss_fun'])+
                       ',regul:{'+str(fgforplot[i]['configuration']['regularizations'])+','+
                        str(fgforplot[i]['configuration']['rlambda'])+'}'+
                      ',n:'+str(fgforplot[i]['configuration']['neurons'])+
                       '\nbs:'+str(fgforplot[i]['configuration']['batch_size'])+
                      ',a1:' + fgforplot[i]['configuration']['activations'][0]+
                        ',a2:' + fgforplot[i]['configuration']['activations'][1]+
                      ',{'+fgforplot[i]['configuration']['optimizers'].pprint()+"}",fontsize=6)
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
        #col.legend(loc=3,prop={'size':10})
        i+=1
plt.show()