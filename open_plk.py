import pickle
import numpy as np
from matplotlib import pyplot as plt

def unstability(x):
    '''
    Calculates a measure of the unstability of the learning. More precisely the unstability
    is determined as a sum of the relative increases of the function value over the iterations.
    Refer to the attached report for a more rigorous definition.
    :param x: A 1-dimensional array representing the function value at each iteration.
    :return: The value of the unstability
    '''
    return \
        np.sum(0 if x[i]<=x[i-1] else np.abs(x[i-1]-x[i])/x[i-1] for i in range(1,len(x)))*1e5/(len(x))

def conv_rate(x, eps=10):
    '''
    Calculates a measure of convergence speed. More precisely it is defined in the [0,100] range.
    A value of 100 is achieved when the function' value is within a percent eps of the minimum
    value reached at the first iteration. A value of 0 is achieved when the function' value
    is not within the range of the minimum value until the last iteration.
    :param x: A 1-dimensional array representing the function value at each iteration.
    :param eps: Specifies a the desired distance from the minimum value as a percentage.
    :return: The value of the convergence speed.
    '''
    return 100-np.argmax(x<=np.min(x)*(1+eps/100))*100/(len(x)-1)

with open('adine1518283445.6559472.pkl', 'rb') as handle:
    b = pickle.load(handle)

ind = 15
cvs=[]
unsts=[]
ll = []
data = []
for i in range(0,len(b)):
#print(b[ind]['tr_loss'])
    unst = unstability(b[i]['tr_loss'])
    cv=conv_rate(b[i]['tr_loss'],eps=25)
    cvs.append(cv)
    ll.append(b[i]['tr_loss'][-1])
    unsts.append(unst)
    data.append(  [b[i]['configuration']['optimizers'].pprint(), 
                        b[i]['tr_loss'][-1],    
                            unstability(b[i]['tr_loss']),   
                                conv_rate(b[i]['tr_loss'],eps=25)])

    print(b[i]['configuration']['optimizers'].pprint(),unst,cv)

"for graph"

for i in range(0,len(cvs)):
    if ll[i]>1.5: pass
    else:
        plt.scatter(ll[i],cvs[i])
        plt.annotate(b[i]['configuration']['optimizers'].pprint()[6:-1],(ll[i],cvs[i]),alpha=0.5,size=5)
plt.ylabel("cvs")
plt.xlabel("ll")
plt.show()  
'''
    
"for table"
columns = ('Param', 'final loss', 'unstability', 'conv_rate')
the_table = plt.table(cellText=data,
                      colLabels=columns, loc="upper center")
the_table.auto_set_font_size(False)
the_table.set_fontsize(7)
the_table.scale(1, 1)  # may help
plt.axis("off")
plt.show()
'''