import numpy as np 
from NN_lib.linesearchs import *

'''
:param x: vector 
'''
def householder_vector(x):
	s = -np.sign(x[0]) * np.linalg.norm(x)
	v = x
	v[0] = v [0] - s
	v = v / np.linalg.norm(v)
	return v , s


def QR_fact(a1):
	a = a1.copy()
	shap = np.shape(a)
	#print(shap)
	q = np.eye(shap[0])

	for j in range(0,shap[1]):
		v , s = householder_vector(a[j:,j])
		a[j,j]=s
		a[j+1:,j]=0
		a[j:,j+1:] = np.subtract(a[j:,j+1:],
								 np.dot(v[:,np.newaxis],
										2*np.dot(np.transpose(v),
												 a[j:,j+1:])[np.newaxis,:]))
		q[:,j:] = q[:,j:] - np.dot(q[:,j:],np.dot(v[:,np.newaxis],2*v[:,np.newaxis].transpose()))
	r=a
	return q,r

def LLSQ(X,y):
	#Could use the QR factorization directly, without explicitely
	#calculating the pseudoinverse
	return np.dot(mypinv(X), y)

def mypinv(x):
	q,r = QR_fact(x)
	#Cut away the 0s
	q = q[:,0:r.shape[1]]
	r = r[0:r.shape[1], :]
	#return np.dot(q,np.linalg.inv(r.transpose()))
	return np.dot(np.linalg.inv(r), q.T)

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


a = np.array(([1,2,3],[4,5,6],[7,8,9])).astype('float32')
a = np.random.randn(30,6)
b = np.array([1,2,4]).astype('float32')
b = np.random.randn(30,2)

sol = np.linalg.lstsq(a,b)
print("**")
print('sol',sol[0])
print(LLSQ(a,b))

