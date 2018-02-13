import pickle
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter


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


opts = []

with open('opts/adam1518335106.9898982.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))

with open('opts/adamax1518361193.7807884.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))

with open('opts/conjgrad1518336080.848611.pkl', 'rb') as handle:
    opts.append(pickle.load(handle))

with open('opts/rmsprop1518335195.034857.pkl', 'rb') as handle:
    opts.append(pickle.load(handle))

with open('opts/simpleopt20k1518361023.1400704.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))

with open('opts/adine1518283445.6559472.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))


with open('opts/mome1518420364.732169.pkl', 'rb') as handle:
   opts.append(pickle.load(handle))


data_to_take = 5
best_data = []

ind = 15
cvs = []
unsts = []
ll = []

for b in opts:
    data = []
    for i in range(0, len(b)):
        # print((len(b[i]['tr_loss'])))
        unst = ((unstability(b[i]['tr_loss'])) * 1e5) / (len(b[i]['tr_loss']))
        cv = conv_rate(b[i]['tr_loss'], eps=10)
        cvs.append(cv)
        ll.append(b[i]['tr_loss'][-1])
        unsts.append(unst)
        data.append([b[i]['configuration']['optimizers'].pprint(),
                     b[i]['tr_loss'][-1],
                     unst,
                     cv])

        print(b[i]['configuration']['optimizers'].pprint(), unst, cv)

    data = sorted(data, key=itemgetter(1))
    for j in range(0, min(data_to_take,len(data))):
    #for j in range(0, len(data)):
        best_data.append(data[j])
    print("\n\n")

"for graph"
'''
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
print(data)
best_data = sorted(best_data, key=itemgetter(1))

digits=4
best_data = [[el[0], np.round(el[1],digits), np.round(el[2],digits), np.round(el[3],digits)]  for el in best_data]
columns = ('Param', 'final loss', 'unstability', 'conv_speed')
the_table = plt.table(cellText=best_data,
                      colWidths=[0.25,0.06,0.06,0.06],
                      colLabels=columns, loc="upper center")

the_table.auto_set_font_size(False)
the_table.set_fontsize(7)
the_table.scale(1, 1.4)

'''cellDict = the_table.get_celld()

for i in range(0, len(columns)):
    for j in range(0, len(data)):
        cellDict[(j, i)].set_width(0.1)
'''
plt.axis("off")
plt.show()