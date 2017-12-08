import numpy as np
import itertools
import loss_functions
import optimizer
import NN
def grid_search(dataset, epochs, n_layers, neurons, activations=None,
				regularizations=None, optimizers=None, batch_size=[32], loss_fun=None,
				val_split=0.25, cvfolds=0):

	#Build up grid for grid search
	grid = dict()
	grid['epochs'] = epochs
	grid['neurons'] = neurons
	grid['batch_size'] = batch_size

	if activations==None:
		grid['activations']=[optimizer.activations['linear']]*n_layers
	else:
		grid['activations'] = activations

	if regularizations==None:
		grid['regularizations']=[loss_functions.reguls['L2']]*n_layers
	else:
		grid['regularizations'] = regularizations

	if optimizers==None:
		grid['optimizers']=[optimizer.optimizers['SGD']]
	else:
		grid['optimizers'] = optimizers

	if loss_fun==None:
		grid['loss_fun']=[loss_functions.losses['MSE']]
	else:
		grid['loss_fun'] = loss_fun

	#Generate all possible hyperparameters configurations
	labels, terms = zip(*grid.items())
	for params in [dict(zip(labels, term)) for term in itertools.product(*terms)]:
		nn = NN.NeuralNetwork()
		result = nn.fit(dataset,params['epochs'], params['optimizers'],
			   params['batch_size'], params['loss_fun'], val_split, cvfolds)
		#TODO update fit to account for val_split and cv_folds (only 1 of them can be set)
		#TODO in fit call k_fold_validation, moving the epochs cycle into k_fold_validation
		#TODO save result with the given hyperparameters somewhere to compare later

def k_fold_validation(dataset,fold_size):


	x_list, y_list = dataset.split_train_k(fold_size)

	for i in range(0,len(x_list)):
		print("---------",i)
		if i == 0:
			train_x = np.concatenate( x_list[i+1:])	
			train_y = np.concatenate( y_list[i+1:])	
		elif i == len(x_list)-1:
			train_x = np.concatenate( x_list[:i]  )	
			train_y = np.concatenate( y_list[:i]  )	
		else:
			train_x = np.concatenate((   x_list[:i]+ x_list[i+1:]    )   )
			train_y = np.concatenate((   y_list[:i]+ y_list[i+1:]    )   )

		test_x  = x_list[i]
		test_y  = y_list[i]
		#TODO cycle epochs as in fit with train_x, train_y then test validation on test_x
	#TODO average over all the folds and return results/best results for the given params
	#TODO fit then basically returns the result of this function
	#TODO peraphs can substitute this with an iterator that generates the folds on demand(dont know how yet)
