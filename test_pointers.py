import numpy as np

history = [{'a':np.array([3,4]), 'b':np.array([6,9])},
           {'a': np.array([7, 6]), 'b': np.array([3, 1])}]


r = {'a':0, 'b':0}
k1 = {d:np.sum(d[k] for k in d.keys()) for d in history}

for d in history:
    for k in d.keys():
        r[k] += d[k]

print(r)
print(k1)