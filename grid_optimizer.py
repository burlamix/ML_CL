
import matplotlib
matplotlib.use('Agg')
from NN_lib import validation
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from NN_lib import preproc
from NN_lib.optimizers import *
import numpy as np
import time
import itertools

np.random.seed(5)
dataset = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"


dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))





adam1 = Adam(lr=0.03,b1=0.9,b2=0.999)
adam2 = Adam(lr=0.4,b1=0.9,b2=0.999)
adam3 = Adam(lr=0.007,b1=0.9,b2=0.999)

adamax1 = Adamax(lr=0.016,b1=0.9,b2=0.999)
adamax2 = Adamax(lr=0.2,b1=0.9,b2=0.999)
adamax3 = Adamax(lr=0.007,b1=0.9,b2=0.999)

RMSP1 = RMSProp(lr=0.01)
RMSP2 = RMSProp(lr=0.05)

cg = ConjugateGradient()


m1 = Momentum(lr=0.1,eps=0.9,nesterov=True)
m2 = Momentum(lr=0.1,eps=0.9,nesterov=True)
m3 = Momentum(lr=0.04,eps=0.9,nesterov=True)


opt_list =[]

opti = dict()
'''
mome
opti["lr"] = [0.1,0.04,0.01,0.005,0.001,0.0003]
opti["eps"] = [0.9,0.6,0.3,0]
opti["nest"] = [True,False]
opt_list.append(Momentum(lr=param["lr"],eps=param["eps"],nesterov=param["nest"]))

adam / adamax
opti["lr"] = [0.4,0.03,0.007,0.0007,]
opti["b1"] = [0.9,0.6,0.3,0]
opti["b2"] = [0.999,0.9,0.8]
opt_list.append(Adam(lr=param["lr"],b1=param["b1"],b2=param["b2"]))


RMSProp
opti["lr"] = [0.2,0.1,0.05,0.01,0.007,0.004,0.0008,0.0002]
opti["delta"] = [0.9,0.8,0.6,0.5,0.3,0.1]
opt_list.append(RMSProp(lr=param["lr"],delta=param["delta"))

adine

opti["lr"] = [0.1,0.01,0.002,0.0005]
opti["ms"] = [0.95,0.7,0.55]
opti["mg"] = [1.0001,1.1]
opti["e"] = [1,1.4]
opt_list.append(Adine(lr=param["lr"],mg=param["mg"],ms=param["ms"],e=param["e"]))

'''
opti["lr"] = [0.4,0.03,0.007,0.0007]
opti["b1"] = [0.9,0.6,0.3,0]
opti["b2"] = [0.999,0.9,0.8]


#lr=0.001, ms=0.9, mg=1.0001, e=1.0,

labels, terms = zip(*opti.items())
all_comb = [dict(zip(labels, term)) for term in itertools.product(*terms)]

for param in all_comb:
    print(param)
    opt_list.append(Adam(lr=param["lr"], b1=param["b1"], b2=param["b2"]))
comb_of_param = len(all_comb)

preprocessor = preproc.Preprocessor()


preprocessor.shuffle(dataset)
acts=[["tanh","linear"]]
opts=opt_list
neurs=[[50,2]]
batches = [dataset.train[0].shape[0]]
losses = ["mee"]
regs = [[regularizations.reguls["EN"],regularizations.reguls["EN"]]]
rlambdas = [[(0.0001,0),(0.0001,0)]]


fgs = list()
start = time.time()

trials = 13

'''
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[3], batch_size=batches,
                                               n_layers=2, val_split=0,activations=acts,
                                               regularizations=regs, rlambda=rlambdas,
                                               cvfolds=1, val_set=None, verbose=0,
                                               loss_fun=losses, val_loss_fun="mee",
                                               neurons=neurs, optimizers=opts,seed=i)
    fgs.append(fg)
'''
fgs = validation.grid_thread(dataset, epochs=[30000], batch_size=batches,
                                           n_layers=2, val_split=0,activations=acts,
                                           regularizations=regs, rlambda=rlambdas,
                                           cvfolds=1, val_set=None, verbose=1,
                                           loss_fun=losses, val_loss_fun="mee",
                                           neurons=neurs, optimizers=opts,trials=trials)



end = time.time()
print('time:', (end-start))
fgmean = list() #List for holding means




#Create initial configs
for i in fgs[0]:
    fgmean.append({'configuration':i['configuration'], 'val_acc':[], 'val_loss':[],
                   'tr_loss':[], 'tr_acc':[]})


for fullgrid in fgs:
    for i in fullgrid:
        for j in range(0,len(fgmean)):
            if i['configuration']==fgmean[j]['configuration']:
                if fgmean[j]['tr_loss']!=[]:
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
    #fgmean[i]['val_acc']/=trials
    fgmean[i]['val_loss']/=trials
    #fgmean[i]['tr_acc']/=trials
    fgmean[i]['tr_loss']/=trials

with open('adam.pkl', 'wb') as output:
    pickle.dump(fgmean, output, pickle.HIGHEST_PROTOCOL)
    #pickle.dump(grid_res, output, pickle.HIGHEST_PROTOCOL)

pp = PdfPages(str(time.time())+".pdf")
plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0-22",ha= "center",va="center", fontsize=40)
pp.savefig()
step1=2
step2=3
step = step1*step2
i=0
for att in range(0,len(opts),step):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=step1*len(batches) * len(neurs) * len(acts),
                          ncols=step2*len(rlambdas) * len(losses),
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['tr_loss'][-1])
    temp = fgforplot[i: i + (step*len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['tr_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            col.set_yticks(np.arange(0, 22, 0.5))
            col.set_title('{'+temp[j]['configuration']['optimizers'].pprint()+"}\n "
                          "last_f:"+str(temp[j]['tr_loss'][-1]),fontsize=20)

            if hist:
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
            else:
                col.plot(temp[j]['tr_loss'], label='tr err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=13)
            col.yaxis.grid()  # horizontal lines
            col.set_ylim([0,22])
            j+=1
            i+=1

    pp.savefig(f)

plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0-3",ha= "center",va="center", fontsize=50)
pp.savefig()
i=0
for att in range(0,len(opts),step):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=step1*len(batches) * len(neurs) * len(acts),
                          ncols=step2*len(rlambdas) * len(losses),
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['tr_loss'][-1])
    temp = fgforplot[i: i + (step*len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['tr_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            col.set_yticks(np.arange(0, 3, 0.07))
            col.set_title('{'+temp[j]['configuration']['optimizers'].pprint()+"}\n "
                          "last_f:"+str(temp[j]['tr_loss'][-1]),fontsize=20)

            if hist:
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
            else:
                col.plot(temp[j]['tr_loss'], label='tr err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=13)
            col.yaxis.grid()  # horizontal lines
            col.set_ylim([0,3])
            j+=1
            i+=1

    pp.savefig(f)

plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0-0.3",ha= "center",va="center", fontsize=50)
pp.savefig()
i=0
for att in range(0,len(opts),step):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=step1*len(batches) * len(neurs) * len(acts),
                          ncols=step2*len(rlambdas) * len(losses),
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['tr_loss'][-1])
    temp = fgforplot[i: i + (step*len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['tr_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            col.set_yticks(np.arange(0, 0.3, 0.0075))
            col.set_title('{'+temp[j]['configuration']['optimizers'].pprint()+"}\n "
                          "last_f:"+str(temp[j]['tr_loss'][-1]),fontsize=20)

            if hist:
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
            else:
                col.plot(temp[j]['tr_loss'], label='tr err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=13)
            col.yaxis.grid()  # horizontal lines
            col.set_ylim([0,0.3])
            j+=1
            i+=1

    pp.savefig(f)
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()
pp.close()
