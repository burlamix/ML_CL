import pickle
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
from conv_measures import unstability, conv_speed
import matplotlib.cm as cm


#-1 to consder all combinations of parameters, x to consider the best x parameters
# configurations based on the final function value achieved
DATA_TO_TAKE = 5

#1 to display a table, 0 for a 2d plot
TABLE_GRAPH = 1

#Sort data by last function value (change to 2 for gen_err, 3 for unstability, 4 for conv spd)
SORT_CRIT = 2

#List of files containing the optimizers history (as output-ed by gird_optimizer.py)
opts_name = ['opts/adam1518335106.9898982.pkl','opts/adamax1518361193.7807884.pkl','opts/rmsprop1518335195.034857.pkl',
        'opts/adine100k1518334988.7234812.pkl','opts/mome1518420364.732169.pkl' ]

#opts_name = [ 'opts/conjgrad1518336080.848611.pkl' ]

opts = []

for name_file in opts_name:
    with open(name_file, 'rb') as handle:
        opts.append(pickle.load(handle))

best_data = []
points_x =[[]]
points_y =[[]]
cvs = []
unsts = []
ll = []
k=0
for b in opts: #For each optimizer
    data = []

    for i in range(0, len(b)): #For each configuration
        unst = unstability(b[i]['tr_loss'])
        cv = conv_speed(b[i]['tr_loss'], eps=10)

        cvs.append(cv)
        unsts.append(unst)

        points_x[k].append(unst)
        points_y[k].append(b[i]['tr_loss'][-1])

        ll.append(b[i]['tr_loss'][-1])
        data.append([b[i]['configuration']['optimizers'].pprint(),
                     b[i]['tr_loss'][-1],
                     b[i]['prediction'][0],
                     unst,
                     cv])

        print(b[i]['configuration']['optimizers'].pprint(), unst, cv)
    points_y.append([])
    points_x.append([])

    data = sorted(data, key=itemgetter(SORT_CRIT))

    if (DATA_TO_TAKE!=-1):
        to=min(DATA_TO_TAKE,len(data))
    else:
        to=len(data)

    for j in range(0, to):
        best_data.append(data[j])
    k=k+1



if (TABLE_GRAPH==0):
    "for graph"

    k=0
    colors = iter(cm.rainbow(np.linspace(0, 1, len(opts))))
    legend=[]

    for b in opts:
        colorr=next(colors)
        legend.append(colorr)
        plt.scatter(points_y[k][0],points_x[k][0],color=colorr,label=b[i]['configuration']['optimizers'].pprint())
        for i in range(0,len(b)):
            if ll[i]>1.5: pass
            else:
                plt.scatter(points_y[k][i],points_x[k][i],color=colorr)
                #plt.annotate(b[i]['configuration']['optimizers'].pprint()[6:-1],(ll[i],cvs[i]),alpha=0.5,size=5)
        k=k+1
    plt.ylabel("unstability")
    plt.xlabel("last function value")

    plt.legend()
    plt.show()  

else:
    "for table"
    best_data = sorted(best_data, key=itemgetter(SORT_CRIT))

    digits=4
    best_data = [[el[0], np.round(el[1],digits), np.round(el[2],digits), np.round(el[3],digits), np.round(el[4],digits), np.round(el[1]/el[2],digits)]  for el in best_data]
    columns = ('Param', 'final loss','gen_err', 'unstability', 'conv_speed','R')
    the_table = plt.table(cellText=best_data,
                          colWidths=[0.40,0.06,0.06,0.06,0.06,0.06],
                          colLabels=columns, loc="upper center")

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)
    the_table.scale(1, 1.2)

    plt.axis("off")
    plt.show()
