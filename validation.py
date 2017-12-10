import numpy as np
import itertools
import loss_functions
import optimizer
from optimizer import *
import NN
import preproc
import sys

#TODO might be useful adding lambda to hyperparameters

#Object with the best hyperparameters and the NN built and trained with them
class grid_result:

	def __init__(self,result_matrix, epochs, batch_size, neurons,activations,optimizer,loss_fun,regularization,n_layers,dataset):

		self.neurons=neurons
		self.activations=activations
		self.optimizer=optimizer
		self.loss_fun=loss_fun
		self.regularization=regularization
		self.result_matrix=result_matrix
		self.epochs = epochs
		self.batch_size = batch_size

		#build neural network
		net = NN.NeuralNetwork()

		in_l = dataset.train[0].shape[1]

		#building neural network
		#	TODO same neurons for all layers  mmmmm.... i dont' like that
		for i in range(0,n_layers):

			if(i!=n_layers-1):
				net.addLayer(in_l,neurons[i],activations[i],regularization=regularization[i])
				in_l=neurons[i]
			else:
				net.addLayer(in_l,dataset.train[1].shape[1],activations[i],regularization=regularization[i])
		net.fit_ds(dataset, self.epochs, self.optimizer, self.batch_size, self.loss_fun)
		self.NN = net 

	#--Not needed
	#you can't chose the optimizer and loss, because grid search chose it for you!
	#def fit(self, dataset, epochs, batch_size):
	#
	#	return self.NN.fit(dataset, epochs, self.optimizer, batch_size, self.loss_fun)

	#--Not needed
	#def evaluate(self,dataset):
	#	return self.NN.evaluate(dataset)




def grid_search(dataset, epochs, n_layers, neurons, activations=None,
				regularizations=None, optimizers=None, batch_size=[32], loss_fun=None, cvfolds=None, val_split=0):

	if(cvfolds==None and val_split==0):cvfolds=3 #If no val selected, default to 3-fold
	if(val_split>0): cvfolds = 1

	if (cvfolds < 1):
		sys.exit("Specify a positive number of folds")
	if (val_split < 0 or val_split >= 100):
		sys.exit("Validation split must be in the range [0,100)")
	if (cvfolds != 1 and val_split > 0):
		sys.exit("Combined cross validation and validation split is currently unsupported")

	validating = False
	if(cvfolds>1 or val_split>0): validating = True
	#Build up grid for grid search
	grid = dict()
	grid['epochs'] = epochs		
	grid['neurons'] = neurons
	grid['batch_size'] = batch_size


	if activations==None:
		grid['activations']=[[optimizer.activations['linear']]*n_layers]
	else:
		grid['activations'] = activations

	if regularizations==None:
		grid['regularizations']=[[loss_functions.reguls['L2']]*n_layers]
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
	#generate list to iterate on
	all_comb = [dict(zip(labels, term)) for term in itertools.product(*terms)]

	#calculate number of configurations
	comb_of_param = len( all_comb )

	#setup matrix to store results (an array is enough if cvfolds returns the avg)
	#result_grid=np.zeros((comb_of_param,cvfolds))
	result_grid = np.zeros(comb_of_param)

	#This list will contain the results for each configuration
	full_grid = []

	k=0
	for params in all_comb :
		net = NN.NeuralNetwork()
		in_l = dataset.train[0].shape[1]
		#building neural network
		for i in range(0,n_layers):
				net.addLayer(in_l,params['neurons'][i],params['activations'][i],regularization=params['regularizations'][i])
				in_l=params['neurons'][i]

		if (val_split > 0 or cvfolds<=1):#Check what kind of validation should be performed
			r = net.fit_ds(dataset, epochs=params['epochs'],val_split=val_split,	\
				optimizer=params['optimizers'],batch_size=params['batch_size'],loss_func=params['loss_fun'])
			if r[1]==None: result_grid[k] = r[0] #If no validation (val_split=0) then select best based on tr loss
			else: result_grid[k] = 1
		else:
			result_grid[k] = k_fold_validation(dataset,cvfolds,net,epochs=params['epochs'],	\
				optimizer=params['optimizers'],batch_size=params['batch_size'],loss_func=params['loss_fun'])

		if (validating):
			full_grid.append({'configuration': params, 'val_loss':result_grid[k]})
		else:#If no validation was done only put config and other stuff(to add..)
			full_grid.append({'configuration': params})
		k=k+1

	#result_avg = np.average(result_grid,axis=1)
	min = np.amin(result_grid)
	ind=np.where(result_grid==min)[0][0]

	best_hyper_param = all_comb[ind]

	best_config = grid_result(result_grid, best_hyper_param['epochs'], best_hyper_param['batch_size'], best_hyper_param['neurons'],best_hyper_param['activations'],\
				best_hyper_param['optimizers'],best_hyper_param['loss_fun'],best_hyper_param['regularizations'],n_layers,dataset)

	prediction = best_config.NN.predict(dataset.test[0]) #Prediction on test
	return full_grid, best_config, prediction

						
def k_fold_validation(dataset,fold_size,NN, epochs, optimizer, batch_size, loss_func ):


	x_list, y_list = dataset.split_train_k(fold_size)

	#dataset for validation,
	#dataset_cv = preproc.Dataset()

	#array of result 
	result = np.zeros((fold_size)) 

	for i in range(0,len(x_list)):
		# make the new test- set and train-set

		#initialize random weights
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

		#dataset_cv.init_train([train_x, train_y])
		#dataset_cv.init_test ([validation_x,  validation_y ])

		#train the model
		NN.fit(train_x, train_y, epochs, optimizer, batch_size, loss_func, verbose=0)
		#NN.fit(dataset_cv, epochs, optimizer, batch_size, loss_func)

		#test the model
		#sarebbe stato più elegante con una tupla, ma cosi facendo quando la passiamo al grid search possiamo concatenare tutto con stack e ottenere una matrice dove basta sommare su un determinato asse..
		#result[i] = NN.evaluate(dataset_cv)
		result[i],acc = NN.evaluate(validation_x, validation_y)
	#TODO return more stuff: in-fold variance, training loss, ..
	return np.average(result)


