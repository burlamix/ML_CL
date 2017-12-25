import numpy
from scipy.spatial import distance


def dist(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2,axis=1))

a = numpy.array([[2, 1],[1,1]])
b = numpy.array([[4, -4],[1,1]])
d = dist(a,b)

print(d)
print(numpy.sqrt(32))
print(numpy.sign(a))

print(numpy.sqrt(b/a))