# Structure of the library


The core of the structure is inside of `NN_lib`. External files show sample usages and experiments on different tasks.


## **NN_lib:**
* **optimizer:** Contains the implementation of the different optimization algorithms (e.g. SGD, Momentum, Adam, Conjugate gradient, ..)
* **linesearches:** Contains the implementation of the backtracking and Armijo-Wolfe line searches
* **NN:** Contains the main functions to work create and use a neural network
* **layer:** Contains the functions concerning the creation and manipulation of a single layer of a neural network
* **activations:** Contains the implementation of a number of activation functions that can be used on the neural network' layers
* **regularization:** Contains the implementation of a number of regularization methods that can be used on the neural network' layers
* **loss_functions:** Contains the implementation of a number of loss functions that can be used for training a neural network
* **validation:** Implements a number of methods such as cross validation and grid search for validating and exploring the parameters of a neural network model

Custom activations, regularizations and loss_functions may be defined, provided they implement the required interface.

 





## **External files:**
* **clean_keras_comp:**  Compares keras models with NN_lib models using the same parameters over randomly generated problems. Note that keras is required to run this file
* **monk_benchmark:** Compares keras with NN_lib on the [MONK problems](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems), a historical set of problems for comparing learning algorithms
* **digits_main:** Compares a keras model with a NN_lib one on a digit recognition task using the [MNIST digits dataset](http://yann.lecun.com/exdb/mnist/)
* **splitData:** Utility file for separating the training data for the ML cup dataset into two separate files for training and testing
* **grid_search_cup:** Contains the template to quickly validate a variety of models on the [ML cup challange](http://pages.di.unipi.it/micheli/DID/CUP-AA1/2017/data2017.html)
* **grid_optimizer:** Contains the template to quickly create, run and plot the behaviour of an optimization algorithms over a wide range of paramaters
* **test_functions:** Contains the implementation of a number of 2-variables test functions for testing optimization algorithms 
* **optimize_2d:** Contains the template to easily visually assess the behaviour of different optimization algorithms	 on `test_functions`
* **llss:** Contains the implementation of linear least squares solver through the QR method and the normal equations one
* **llss_nn:** Contains the template for quickly comparing neural network, `llss` linear least squares solvers and `numpy`' linear least squares solvers on the ML cup dataset. The generlization error of the least squares solvers is calculated as well
* **open_plk:** Contains the implementation of two rough measures of convergence speed and unstability given an array whose elements represent the value of a function over the iterations
