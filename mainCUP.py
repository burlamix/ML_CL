import preproc
from NN_lib import validation
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle

np.random.seed(5)
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
optimizer5 = Momentum(lr=0.5,eps=0.5)
optimizer10 = RMSProp(lr=0.0151)

#adam1 = Adamax(lr=0.011)

adam1 = Adam(lr=0.03,b1=0.9,b2=0.999)
adam2 = Adam(lr=0.4,b1=0.9,b2=0.999)
adam3 = Adam(lr=0.007,b1=0.9,b2=0.999)

adamax1 = Adamax(lr=0.016,b1=0.9,b2=0.999)

adamax2 = Adamax(lr=0.2,b1=0.9,b2=0.999)
adamax3 = Adamax(lr=0.007,b1=0.9,b2=0.999)

RMSP1 = RMSProp(lr=0.01)


m1 = Momentum(lr=0.001,eps=0.9,nesterov=True)
m2 = Momentum(lr=0.008,eps=0.9,nesterov=True)
m3 = Momentum(lr=0.04,eps=0.9,nesterov=True)

#no m3 = Momentum(lr=0.3,eps=0.9,nesterov=True)

preprocessor = preproc.Preprocessor()
#preprocessor.remove_outliers(dataset)
#print(preprocessor.remove_outliers(dataset,sensitivity=2.5))

#preprocessor.outlier_range(dataset,0.001,2,3)
#exit(1)

preprocessor.shuffle(dataset)
#preprocessor.normalize(dataset,method='minmax')
acts=[["sigmoid","linear"]]
opts=[adamax1]
neurs=[[50,2]]
batches = [dataset.train[0].shape[0]]
losses = ["mee"]
regs = [[regularizations.reguls["EN"],regularizations.reguls["EN"]]]
rlambdas = [[(0.0001,0.0001),(0.0001,0.0001)]]
           # [(0.001), (0,0)]]

fgs = list()

trials = 1
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[3105], batch_size=batches,
                                               n_layers=2, val_split=0,activations=acts,
                                               regularizations=regs, rlambda=rlambdas,
                                               cvfolds=1, val_set=None, verbose=1,
                                               loss_fun=losses, val_loss_fun="mee",
                                               neurons=neurs, optimizers=opts)
    fgs.append(fg)
#print(grid_res.NN.evaluate(dataset2016.train[0],dataset2016.train[1],"mee"))
#exit(1)
fgmean = list() #List for holding means

with open('rimuovendo_out.pkl', 'wb') as output:
    pickle.dump(fgs, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(grid_res, output, pickle.HIGHEST_PROTOCOL)


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

pp = PdfPages("rimuovendo_out.pdf")
#nconfig = len(acts)*len(opts)*len(neurs)
#TODO MEDIA SU PIu test vedi k validation


i=0

for att in opts:
    f, (a) = plt.subplots(figsize=(30, 30), nrows=len(batches) * len(neurs) * len(acts),
                          ncols=len(rlambdas) * len(losses),
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['val_loss'][-1])
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
                #col.plot(fgforplot[i]['history']['tr_acc'],label='tr acc',marker='1')
                #col.plot(fgforplot[i]['history']['val_acc'],label='val acc')
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
                col.plot(temp[j]['history']['val_loss'],label='val err')
            else:
                #col.plot(fgforplot[i]['tr_acc'], label='tr acc',ls=":")
                #col.plot(fgforplot[i]['val_acc'], label='val acc',ls="--")
                col.plot(temp[j]['tr_loss'], label='tr err',ls='-.')
                col.plot(temp[j]['val_loss'], label='val err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=6)
            col.set_ylim([0,22])
            j+=1
            i+=1
    #plt.show()

    pp.savefig(f)
plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0-3",ha= "center",va="center")
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()
i=0

for att in opts:

    f, (a) = plt.subplots(figsize=(30,30),nrows=len(batches)*len(neurs)*len(acts), ncols=len(rlambdas)*len(losses), sharex='col', sharey='row',squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())

    temp = fgforplot[i: i + (len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['val_loss'][-1])
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
                #col.plot(fgforplot[i]['history']['tr_acc'],label='tr acc',marker='1')
                #col.plot(fgforplot[i]['history']['val_acc'],label='val acc')
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
                col.plot(temp[j]['history']['val_loss'],label='val err')
            else:
                #col.plot(fgforplot[i]['tr_acc'], label='tr acc',ls=":")
                #col.plot(fgforplot[i]['val_acc'], label='val acc',ls="--")
                col.plot(temp[j]['tr_loss'], label='tr err',ls='-.')
                col.plot(temp[j]['val_loss'], label='val err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=6)
            col.set_ylim([0,3])
            j+=1
            i+=1
    #plt.show()

    pp.savefig(f)

plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0.5-1.2",ha= "center",va="center")
pp.savefig()
f.clear()
plt.clf()
f.clf()
plt.cla()
plt.close()


i=0

for att in opts:

    f, (a) = plt.subplots(figsize=(30,30),nrows=len(batches)*len(neurs)*len(acts), ncols=len(rlambdas)*len(losses), sharex='col', sharey='row',squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())

    temp = fgforplot[i: i + (len(batches)*len(neurs)*len(acts)*len(rlambdas)*len(losses))]
    #temp = sorted(temp, key=lambda k:k['val_loss'][-1])
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
                #col.plot(fgforplot[i]['history']['tr_acc'],label='tr acc',marker='1')
                #col.plot(fgforplot[i]['history']['val_acc'],label='val acc')
                col.plot(temp[j]['history']['tr_loss'],label='tr err')
                col.plot(temp[j]['history']['val_loss'],label='val err')
            else:
                #col.plot(fgforplot[i]['tr_acc'], label='tr acc',ls=":")
                #col.plot(fgforplot[i]['val_acc'], label='val acc',ls="--")
                col.plot(temp[j]['tr_loss'], label='tr err',ls='-.')
                col.plot(temp[j]['val_loss'], label='val err')
            #col.legend(loc=3,prop={'size':10})
            col.tick_params(labelsize=6)
            col.set_ylim([0.75,1.2])
            j+=1
            i+=1
    #plt.show()

    pp.savefig(f)
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()


pp.close()