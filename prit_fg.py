
import preproc
from NN_lib import validation
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle



pp = PdfPages("hope.pdf")



with open('sceaning.pkl', 'rb') as inputt:
     fgs=pickle.load(inputt)

fgmean = list() #List for holding means

#Create initial configs
for i in fgs[0]:
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
    fgmean[i]['val_acc']/=len(fgs)
    fgmean[i]['val_loss']/=len(fgs)
    fgmean[i]['tr_acc']/=len(fgs)
    fgmean[i]['tr_loss']/=len(fgs)
#grid 

print(len(fgmean))

i=0
for a in range(0,9):
    f, (a) = plt.subplots(figsize=(30, 30), nrows=9,
                          ncols=4,
                          sharex='col', sharey='row', squeeze=False)
    fgforplot=fgmean
    fgforplot=sorted(fgforplot,key=lambda k:k['configuration']['optimizers'].__str__())
    temp = fgforplot[i: i + (36)]
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
                          ',{'+temp[j]['configuration']['optimizers'].pprint()+"}"+
                          ",{FINALE}"+str(temp[j]["val_loss"][-1]),fontsize=10)

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
plt.text(0.5,0.5,"range 0-3",ha= "center",va="center")
pp.savefig()
f.clear()
f.clf()
plt.clf()
plt.cla()
plt.close()
pp.close()