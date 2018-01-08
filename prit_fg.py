
import preproc
from NN_lib import validation
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle



pp = PdfPages("momgrids.pdf")


with open('momgrids.pkl', 'rb') as inputt:
     fgs=pickle.load(inputt)

fgmean = list() #List for holding means

#Create initial configs
for i in fgs[0]:
    fgmean.append({'configuration':i['configuration'], 'val_acc':[], 'val_loss':[],"in-fold var":0,
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
                    fgmean[j]['in-fold var']+=np.array(i['in-fold var'])
                else:
                    fgmean[j]['val_acc']=np.array(i['history']['val_acc'])
                    fgmean[j]['val_loss']=np.array(i['history']['val_loss'])
                    fgmean[j]['tr_acc']=np.array(i['history']['tr_acc'])
                    fgmean[j]['tr_loss']=np.array(i['history']['tr_loss'])
                    fgmean[j]['in-fold var']=np.array(i['in-fold var'])
                break

for i in range(0,len(fgmean)):
    fgmean[i]['val_acc']/=len(fgs)
    fgmean[i]['val_loss']/=len(fgs)
    fgmean[i]['tr_acc']/=len(fgs)
    fgmean[i]['tr_loss']/=len(fgs)
    fgmean[i]['in-fold var']/=len(fgs)
#grid 

print("-----",fgmean[0]['in-fold var'])

i=0
for a in range(0,1):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=6,
                          ncols=5,
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (36)]
    temp = sorted(temp, key=lambda k:k['val_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            print(j)
            col.set_title('tl:'+str(temp[j]['configuration']['loss_fun'])+
                          ',vl:'+str(temp[j]['configuration']['val_loss_fun'])+
                           ',rg:{'+str(temp[j]['configuration']['regularizations'])+','+
                            str(temp[j]['configuration']['rlambda'])+'}'+
                          '\nn:'+str(temp[j]['configuration']['neurons'])+
                           ',bs:'+str(temp[j]['configuration']['batch_size'])+
                          ',a1:' + temp[j]['configuration']['activations'][0]+
                            ',a2:' + temp[j]['configuration']['activations'][1]+
                          ',{'+temp[j]['configuration']['optimizers'].pprint()+"}\n"+
                          "{val_loss="+str(temp[j]["val_loss"][-1])+"}"+
                          ",{tr_loss="+str(temp[j]["tr_loss"][-1])+"}"+
                          "\nin-fold var"+str(temp[j]["in-fold var"])+
                          ".",fontsize=10)

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
    pp.savefig(f)
plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0.75-1.2",ha= "center",va="center")
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()



i=0
for a in range(0,1):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=6,
                          ncols=5,
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (30)]
    temp = sorted(temp, key=lambda k:k['val_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            print(j)
            col.set_title('tl:'+str(temp[j]['configuration']['loss_fun'])+
                          ',vl:'+str(temp[j]['configuration']['val_loss_fun'])+
                           ',rg:{'+str(temp[j]['configuration']['regularizations'])+','+
                            str(temp[j]['configuration']['rlambda'])+'}'+
                          '\nn:'+str(temp[j]['configuration']['neurons'])+
                           ',bs:'+str(temp[j]['configuration']['batch_size'])+
                          ',a1:' + temp[j]['configuration']['activations'][0]+
                            ',a2:' + temp[j]['configuration']['activations'][1]+
                          ',{'+temp[j]['configuration']['optimizers'].pprint()+"}\n"+
                          "{val_loss="+str(temp[j]["val_loss"][-1])+"}"+
                          ",{tr_loss="+str(temp[j]["tr_loss"][-1])+"}"+
                          "\nin-fold var"+str(temp[j]["in-fold var"])+
                          ".",fontsize=10)

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
    pp.savefig(f)
plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0.75-1.2",ha= "center",va="center")
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()
print("--")


i=0
for a in range(0,1):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=6,
                          ncols=5,
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (30)]
    temp = sorted(temp, key=lambda k:k['val_loss'][-1])
    j=0
    hist=False
    for row in a:
        for col in row:
            print(j)
            col.set_title('tl:'+str(temp[j]['configuration']['loss_fun'])+
                          ',vl:'+str(temp[j]['configuration']['val_loss_fun'])+
                           ',rg:{'+str(temp[j]['configuration']['regularizations'])+','+
                            str(temp[j]['configuration']['rlambda'])+'}'+
                          '\nn:'+str(temp[j]['configuration']['neurons'])+
                           ',bs:'+str(temp[j]['configuration']['batch_size'])+
                          ',a1:' + temp[j]['configuration']['activations'][0]+
                            ',a2:' + temp[j]['configuration']['activations'][1]+
                          ',{'+temp[j]['configuration']['optimizers'].pprint()+"}\n"+
                          "{val_loss="+str(temp[j]["val_loss"][-1])+"}"+
                          ",{tr_loss="+str(temp[j]["tr_loss"][-1])+"}"+
                          "\nin-fold var"+str(temp[j]["in-fold var"])+
                          ".",fontsize=10)

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
    pp.savefig(f)
plt.figure()
plt.axis("off")
plt.text(0.5,0.5,"range 0.75-1.2",ha= "center",va="center")
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()

print(np.average([i["in-fold var"] for i in fgmean]))



pp.close()
