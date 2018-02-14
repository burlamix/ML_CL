import numpy as np 


def householder_vector(x):
	s = -np.sign(x[0]) * np.linalg.norm(x)
	v = x
	v[0] = v [0] - s
	v = v / np.linalg.norm(v)
	return v , s


def QR_fact(X):
	''''
	QR factorization via householder reflectors.
	'''
	a = X.copy()
	shap = np.shape(a)
	q = np.eye(shap[0])

	for j in range(0,shap[1]):
		v , s = householder_vector(a[j:,j])
		a[j,j]=s
		a[j+1:,j]=0
		a[j:,j+1:] = np.subtract(a[j:,j+1:],
								 np.dot(v[:,np.newaxis],
										2*np.dot(np.transpose(v),
												 a[j:,j+1:])[np.newaxis,:]))
		q[:,j:] = q[:,j:] - np.dot(q[:,j:],
								   np.dot(v[:,np.newaxis],
										  2*v[:,np.newaxis].transpose()))
	r=a
	return q,r


def QR_solver(X,y,l=0):
	'''
	Linear least squares solver via QR decomposition. It solves a problem of
	the form min( || Ax-y+r||x|| || ). It is assumed that A is of full column
	rank. Refer to the report for details on how the solution is obtained.
	:param l: regularization term of L2.
	'''
	X = np.vstack((X,l*np.eye(11)))
	q, r = QR_fact(X)

	#Throw away the 0s
	q1 = q[:-11, 0:r.shape[1]]
	r = r[0:r.shape[1], :]
	return np.dot(np.dot(np.linalg.inv(r),q1.T),y)


def norm_eqs(X,y,l=0):
	'''
	Linear least squares solver via normal equations. It solves a problem of
	the form min( || Ax-y+r||x|| || ). It is assumed that A is of full column
	rank. Refer to the report for details on how the solution is obtained.
	:param l: regularization term of L2.
	'''
	reg = np.eye(X.shape[1])*l**2
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+reg),X.T), y)


def mypinv(X):
	'''
	Computes the pseudoinverse of the input matrix X via the QR
	decomposition. Refer to the report for details on how the solution
	is obtained.
	'''
	q,r = QR_fact(X)
	#Throw away the 0s
	q = q[:,0:r.shape[1]]
	r = r[0:r.shape[1], :]
	return np.dot(np.linalg.inv(r), q.T)

