import pickle
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import matplotlib.cm as cm

def unstability(x):
    '''
    Calculates a measure of the unstability of the learning. More precisely the unstability
    is determined as a sum of the relative increases of the function value over the iterations.
    Refer to the attached report for a more rigorous definition.
    :param x: A 1-dimensional array representing the function value at each iteration.
    :return: The value of the unstability
    '''
    return \
        np.sum(0 if x[i] <= x[i - 1] else np.abs(x[i - 1] - x[i]) / x[i - 1] for i in range(1, len(x)))


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
    return 100 - np.argmax(x <= np.min(x) * (1 + eps / 100)) * 100 / (len(x) - 1)


#list of file to upload
opts = []

#file to upload
with open('opts/adam1518335106.9898982.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))

with open('opts/adamax1518361193.7807884.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))

#with open('opts/conjgrad1518336080.848611.pkl', 'rb') as handle:
#    opts.append(pickle.load(handle))

with open('opts/rmsprop1518335195.034857.pkl', 'rb') as handle:
    opts.append(pickle.load(handle))

#with open('opts/simpleopt20k1518361023.1400704.pkl', 'rb') as handle:
#   opts.append(pickle.load(handle))

with open('opts/adine100k1518334988.7234812.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))


with open('opts/mome1518420364.732169.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))


#-1 if you want take all combination of parameter, x if you want take maximun the best parameter configuration 
data_to_take = -1

# put 1 if you want a table, 0 if you want a 2d plots
table_graph=1



best_data = []
points_x =[[]]
points_y =[[]]
ind = 15
cvs = []
unsts = []
ll = []
k=0
for b in opts:
    data = []

    for i in range(0, len(b)):
        unst = ((unstability(b[i]['tr_loss'])) * 1e5) / (len(b[i]['tr_loss']))
        cv = conv_rate(b[i]['tr_loss'], eps=10)
        cvs.append(cv)
        
        points_x[k].append(cv)
        points_y[k].append(b[i]['tr_loss'][-1])

        ll.append(b[i]['tr_loss'][-1])
        unsts.append(unst)
        data.append([b[i]['configuration']['optimizers'].pprint(),
                     b[i]['tr_loss'][-1],
                     b[i]['prediction'][0],
                     unst,
                     cv])

        print(b[i]['configuration']['optimizers'].pprint(), unst, cv)
    points_y.append([])
    points_x.append([])
    data = sorted(data, key=itemgetter(2))

    if (data_to_take==-1):
        to=min(data_to_take,len(data))
    else:
        to=len(data)

    for j in range(0, to):
        best_data.append(data[j])
    print("\n\n")
    k=k+1


"for graph"

if (table_graph==0):
    k=0
    colors = iter(cm.rainbow(np.linspace(0, 1, len(opts))))
    legend=[]

    for b in opts:
        colorr=next(colors)
        legend.append(colorr)
        plt.scatter(points_y[k][0],points_x[k][0],color=colorr,label=b[i]['configuration']['optimizers'].pprint())
        for i in range(0,len(b)):
            print("-"+str(i))
            if ll[i]>1.5: pass
            else:
                plt.scatter(points_y[k][i],points_x[k][i],color=colorr)
                #plt.annotate(b[i]['configuration']['optimizers'].pprint()[6:-1],(ll[i],cvs[i]),alpha=0.5,size=5)
        k=k+1
    plt.ylabel("cvs")
    plt.xlabel("ll")

    plt.legend()
    plt.show()  
else:

    "for table"
    print(data)
    best_data = sorted(best_data, key=itemgetter(2))

    digits=4
    best_data = [[el[0], np.round(el[1],digits), np.round(el[2],digits), np.round(el[3],digits), np.round(el[4],digits)]  for el in best_data]
    columns = ('Param', 'final loss','gen_err', 'unstability', 'conv_speed')
    the_table = plt.table(cellText=best_data,
                          colWidths=[0.40,0.06,0.06,0.06,0.06],
                          colLabels=columns, loc="upper center")

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)
    the_table.scale(1, 1.2)

    plt.axis("off")
    plt.show()