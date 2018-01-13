import numpy as np 
from NN_lib.linesearchs import *

'''
:param x: vector 
'''
def householder_vector(x):
	s = -np.sign(x[0]) * np.linalg.norm(x)
	#print(s)
	v = x
	v[0] = v [0] - s
	#print("v1",v)
	v = v / np.linalg.norm(v)
	#print("v2",v)
	return v , s


def QR_fact(a):
	shap = np.shape(a)
	#print(shap)
	q = np.eye(shap[0])

	for j in range(0,shap[1]):
		v , s = householder_vector(a[j:,j])
		a[j,j]=s
		a[j+1:,j]=0

		a[j:,j+1:] = a[j:,j+1:] - 2* np.dot(np.transpose(v),a[j:,j+1:])
		
		q[:,j:] = q[:,j:] - q[:,j:]*v*2*np.transpose(v)
	
	r=a
	return q,r


''' esempio sulle slide, su mathlab torna -1e+10-0-0-0 e qui -1e10,-1e6,-1e6,-1e6 
ma credo sia dovuto alle approssimazioni già il primo arrya x matlab lo considera 
uno e tutti zero
x= np.array( [1e10, 1e-6, 1e-6, 1e-6])
u , s = householder_vector(x)
r = x - 2*u*(np.transpose(u)*x)
print("u  ",u)
print("s  ",s)
print("r  ",r)
'''


a = np.array(([1,2,3],[4,5,6],[7,8,9]))
b = np.array([1,2,3,4])

#print(householder_vector(b))

print("my",QR_fact(a))

print("np",np.linalg.qr(a))

