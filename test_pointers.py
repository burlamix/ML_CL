import numpy as np

history = [{'a':np.array([3,4]), 'b':np.array([6,9])},
           {'a': np.array([7, 6]), 'b': np.array([3, 1])}]


r = {'a':0, 'b':0}
for d in history:
    for k in d.keys():
        r[k] += d[k]

print(r)