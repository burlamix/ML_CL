import pickle
import numpy as np
from scipy.stats import spearmanr
from conv_measures import unstability, conv_speed

#List of files containing the optimizers history (as output-ed by gird_optimizer.py)
opt_name = 'opts/adam1518335106.9898982.pkl'
#opt_name = 'opts/adamax1518361193.7807884.pkl'
#opt_name = 'opts/rmsprop1518335195.034857.pkl',
#opt_name = 'opts/adine100k1518334988.7234812.pkl'
#opt_name =  'opts/mome1518420364.732169.pkl'

with open(opt_name, 'rb') as handle:
        opt = pickle.load(handle)

#Change momentum if loading a momentum, adine if adine, adam if adam,
# adamax if adamax, rms if rmsprop
opt_type = "adam"

data = []
for i in range(0, len(opt)):
        data.append([opt[i]['configuration']['optimizers'],
                     opt[i]['tr_loss'][-1],
                     unstability(opt[i]['tr_loss']),
                     conv_speed(opt[i]['tr_loss'], eps=10)])



#Lazy parameterization ahead
if opt_type=="adam" or opt_type=="adamax":
    b1=[]; b2=[]

if opt_type == "adine":
    ms=[]; mg=[]

if opt_type == "momentum":
    m=[]; nstv=[]

if opt_type == "rms":
    delta=[] #For RMSprop

lr=[]; ll=[]; unst=[];convspd=[]

for el in data:
    lr.append(el[0].lr)
    ll.append(el[1])
    unst.append(el[2])
    convspd.append(el[3])

    if opt_type == "adam" or opt_type=="adamax":
        b1.append(el[0].b1)
        b2.append(el[0].b2)

    if opt_type=="adine":
        ms.append(el[0].ms)
        mg.append(el[0].mg)

    if opt_type=="momentum":
        m.append(el[0].eps)
        nstv.append(1 if el[0].nesterov==True else 0)

    if opt_type=="rms":
        delta.append(el[0].delta)

params = []
if opt_type=="adam" or opt_type=="adamax":
    print("Order: step size, b1, b2, last val, unstability, conv speed")
    params.append([lr, b1, b2, ll, unst, convspd])

if opt_type=="adine":
    print("Order: step size, minor momentum, greater momentum, last val, unstability, conv speed")
    params.append([lr, ms, mg, ll, unst, convspd])

if opt_type == "momentum":
    print("Order: step size, momentum, nesterov, last val, unstability, conv speed")
    params.append([lr, m, nstv, ll, unst, convspd])

if opt_type == "rms":
    print("Order: step size, delta, last val, unstability, conv speed")
    params.append([lr, delta, ll, unst, convspd])

pearson=np.corrcoef(params[0]).round(4)


print(pearson)
#spm = spearmanr([lr,ms,nstv,ll,unst,convspd],axis=1)[0].round(4)
#print("\n\n\n **********")
#print(spm)

#print(pearson-spm)