import numpy as np

def divide_in_fold(dataset,fold_size): 


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

		#DO GRID SEARCH 

