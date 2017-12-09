import numpy as np
import itertools
import loss_functions
import optimizer
from optimizer import *
import NN
import preproc
import NN


#object with alla data grom grid search
#it has inside a NN build with the best hyperparam find
class grid_result:

	def __init__(self,result_matrix,neurons,activations,optimizer,loss_fun,regularization,n_layers,dataset):

		self.neurons=neurons
		self.activations=activations
		self.optimizer=optimizer
		self.loss_fun=loss_fun
		self.regularization=regularization
		self.result_matrix=result_matrix

		#build neural network
		net = NN.NeuralNetwork()

		in_l = dataset.train[0].shape[1]

		#building neural network
		#	TODO same neurons for all layers  mmmmm.... i dont' like that
		for i in range(0,n_layers):

			if(i!=n_layers-1):
				net.addLayer(in_l,neurons,activations,regularization=regularization)
				in_l=neurons
			else:
				net.addLayer(in_l,dataset.train[1].shape[1],activations,regularization=regularization)

		self.NN = net 

	#you can't chose the optimizer and loss, because grid search chose it for you!
	def fit(self, dataset, epochs, batch_size):

		return self.NN.fit(dataset, epochs, self.optimizer, batch_size, self.loss_fun)


	def evaluate(self,dataset):

		return self.NN.evaluate(dataset)




def grid_search(dataset, epochs, n_layers, neurons, activations=None,
				regularizations=None, optimizers=None, batch_size=[1016], loss_fun=None,
				val_split=0.25, cvfolds=3):

	#matrix of result 

	#Build up grid for grid search
	grid = dict()
	grid['epochs'] = epochs		
	grid['neurons'] = neurons
	grid['batch_size'] = batch_size


	if activations==None:
		grid['activations']=[optimizer.activations['linear']]
		#grid['activations']=[optimizer.activations['linear']]*n_layers   ??
	else:
		grid['activations'] = activations

	if regularizations==None:
		grid['regularizations']=[loss_functions.reguls['L2']]
		#grid['regularizations']=[loss_functions.reguls['L2']]*n_layers   ??
	else:
		grid['regularizations'] = regularizations

	if optimizers==None:
		grid['optimizers']=[optimizer.optimizers['SGD']]
	else:
		grid['optimizers'] = optimizers

	if loss_fun==None:
		grid['loss_fun']=[loss_functions.losses['mse']]
	else:
		grid['loss_fun'] = loss_fun

	#Generate all possible hyperparameters configurations
	labels, terms = zip(*grid.items())
	#generate array where iterate
	all_comb = [dict(zip(labels, term)) for term in itertools.product(*terms)]

	#take number of combination
	comb_of_param = len( all_comb )

	#make matrix for result
	result_grid=np.zeros((comb_of_param,cvfolds))


	k=0
	for params in all_comb :

		net = NN.NeuralNetwork()
		in_l = dataset.train[0].shape[1]

		#building neural network
		#	TODO same neurons for all layers  mmmmm.... i dont' like that
		for i in range(0,n_layers):

			if(i!=n_layers-1):
				net.addLayer(in_l,params['neurons'],params['activations'],regularization=params['regularizations'])
				in_l=params['neurons']
			else:
				net.addLayer(in_l,dataset.train[1].shape[1],params['activations'],regularization=params['regularizations'])

		#make k-fold
		result_grid[k] = k_fold_validation(dataset,cvfolds,net,epochs=params['epochs'],	\
				optimizer=params['optimizers'],batch_size=params['batch_size'],loss_func=params['loss_fun'])

		k=k+1

		#result = nn.fit(dataset,params['epochs'], params['optimizers'],
		#	   params['batch_size'], params['loss_fun'], val_split, cvfolds)
		#TODO update fit to account for val_split and cv_folds (only 1 of them can be set)
		#TODO in fit call k_fold_validation, moving the epochs cycle into k_fold_validation
		#TODO save result with the given hyperparameters somewhere to compare later

	result_avg = np.sum(result_grid,axis=1)	

	min = np.amin(result_avg)
	ind=np.where(result_avg==min)[0][0]

	best_hyper_param = all_comb[ind]
	
	return grid_result(result_avg,best_hyper_param['neurons'],best_hyper_param['activations'],\
				best_hyper_param['optimizers'],best_hyper_param['loss_fun'],best_hyper_param['regularizations'],n_layers,dataset)


						
def k_fold_validation(dataset,fold_size,NN, epochs, optimizer, batch_size, loss_func ):


	x_list, y_list = dataset.split_train_k(fold_size)

	#dataset for validation,
	dataset_cv = preproc.Dataset()

	#array of result 
	result = np.zeros((fold_size)) 

	for i in range(0,len(x_list)):
		# make thw new test- set and train-set

		#inizialire random weight for neural netwkor
		NN.initialize_random_weight()

		if i == 0:
			train_x = np.concatenate( x_list[i+1:])	
			train_y = np.concatenate( y_list[i+1:])	
		elif i == len(x_list)-1:
			train_x = np.concatenate( x_list[:i]  )	
			train_y = np.concatenate( y_list[:i]  )	
		else:
			train_x = np.concatenate((   x_list[:i]+ x_list[i+1:]    )   )
			train_y = np.concatenate((   y_list[:i]+ y_list[i+1:]    )   )

		validation_x  =  x_list[i]
		validation_y  =  y_list[i]

		dataset_cv.init_train([train_x, train_y])
		dataset_cv.init_test ([validation_x,  validation_y ])

		#train the model
		NN.fit(dataset_cv, epochs, optimizer, batch_size, loss_func)

		#test the model
		#sarebbe stato più elegante con una tupla, ma cosi facendo quando la passiamo al grid search possiamo concatenare tutto con stack e ottenere una matrice dove basta sommare su un determinato asse..
		result[i] = NN.evaluate(dataset_cv)

		#TODO cycle epochs as in fit with train_x, train_y then test validation on test_x
	#TODO average over all the folds and return results/best results for the given params	
	#TODO fit then basically returns the result of this function
	#TODO peraphs can substitute this with an iterator that generates the folds on demand(dont know how yet)

	return result


