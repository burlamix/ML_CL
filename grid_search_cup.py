from NN_lib import validation
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from NN_lib import preproc
from NN_lib.optimizers import *
import numpy as np

np.random.seed(5)
dataset = preproc.Dataset()

#Loading the dataset
train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"
dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))

#Creating optimizers to validate on
adam1 = Adam(lr=0.03,b1=0.9,b2=0.999)
adam2 = Adam(lr=0.4,b1=0.9,b2=0.999)
adam3 = Adam(lr=0.007,b1=0.9,b2=0.999)
adamax1 = Adamax(lr=0.016,b1=0.9,b2=0.999)
adamax2 = Adamax(lr=0.2,b1=0.9,b2=0.999)
adamax3 = Adamax(lr=0.007,b1=0.9,b2=0.999)
RMSP1 = RMSProp(lr=0.01)
RMSP2 = RMSProp(lr=0.05)
m1 = Momentum(lr=0.001,eps=0.9,nesterov=True)
m2 = Momentum(lr=0.008,eps=0.9,nesterov=True)
m3 = Momentum(lr=0.04,eps=0.9,nesterov=True)


preprocessor = preproc.Preprocessor()
#Uncomment lines below to remove outliers or normalize data
#preprocessor.remove_outliers(dataset,sensitivity=2.5))
#preprocessor.normalize(dataset,method='minmax')
preprocessor.shuffle(dataset)


#Create the configurations to build models and validate on.
acts=[["sigmoid","linear"],["tanh","linear"],["relu","linear"]]
opts=[adam1,adam2,adam3,adamax1,adamax2,adamax3,RMSP1,RMSP2,m1,m2,m3]
neurs=[[20,2],[50,2],[80,2]]
batches = [dataset.train[0].shape[0]]
losses = ["mee","mae","mse"]
regs = [[regularizations.reguls["EN"],regularizations.reguls["EN"]]]
rlambdas = [[(0.0001,0.0001),(0.000,0.0001)],[(0.0006,0.0004),(0.0006,0.0004)],
[(0.001,0.001),(0.01,0.01)],[(0.000,0.0001),(0.000,0.0001)],[(0.0001,0.000),(0.0001,0.000)]]

fgs = list()

trials = 2
#Validate the models resulting from the combination of the parameters above
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[30], batch_size=batches,
                                               n_layers=2, val_split=0,activations=acts,
                                               regularizations=regs, rlambda=rlambdas,
                                               cvfolds=3, val_set=None, verbose=1,
                                               loss_fun=losses, val_loss_fun="mee",
                                               neurons=neurs, optimizers=opts)
    fgs.append(fg)

fgmean = list() #List for holding means

#Save the results of all configurations -- might eat up some space
with open('grid_save.pkl', 'wb') as output:
    pickle.dump(fgs, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(grid_res, output, pickle.HIGHEST_PROTOCOL)


########### ugly code for generating a pdf with the plots ###########

#Create initial configs
for i in fg:
    fgmean.append({'configuration':i['configuration'], 'val_acc':[], 'val_loss':[],
                   'tr_loss':[], 'tr_acc':[]})


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

#Average over the trials
for i in range(0,len(fgmean)):
    fgmean[i]['val_acc']/=trials
    fgmean[i]['val_loss']/=trials
    fgmean[i]['tr_acc']/=trials
    fgmean[i]['tr_loss']/=trials

pp = PdfPages("grid_plot.pdf")

i=0
for att in opts:
    f, (a) = plt.subplots(figsize=(30, 30), nrows=len(batches) * len(neurs) * len(acts),
                          ncols=len(rlambdas) * len(losses),
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    temp = sorted(temp, key=lambda k:k['val_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            col.set_title('tl:'+str(temp[j]['configuration']['loss_fun'])+
                          ',vl:'+str(temp[j]['configuration']['val_loss_fun'])+
                           ',rg:{'+str(temp[j]['configuration']['regularizations'])+','+
                            str(temp[j]['configuration']['rlambda'])+'}'+
                          '\nn:'+str(temp[j]['configuration']['neurons'])+
                           ',bs:'+str(temp[j]['configuration']['batch_size'])+
                          ',a1:' + temp[j]['configuration']['activations'][0]+
                            ',a2:' + temp[j]['configuration']['activations'][1]+
                          ',{'+temp[j]['configuration']['optimizers'].pprint()+"}",fontsize=10)

            if hist:
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
                col.plot(temp[j]['history']['val_loss'],label='val err')
            else:
                col.plot(temp[j]['tr_loss'], label='tr err',ls='-.')
                col.plot(temp[j]['val_loss'], label='val err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=6)
            col.set_ylim([0,22])
            j+=1
            i+=1

    pp.savefig(f)
plt.figure()
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()
pp.close()