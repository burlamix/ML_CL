A implementation of a neural network library to create, train and validate neural network models. In particular the focus is on the implementation of a number of first-order optimization methods, that is using only the information given by the gradient. For an overview of some of the methods implemented you can refer to the [work](https://arxiv.org/abs/1609.04747) of S. Ruder.

# Structure of the library (WIP)


The core of the structure is inside of `NN_lib`. External files show sample usages and experiments on different tasks.
Note that a version of Python3 is required. Additional keras (and tensorflow(tested) or theano(not tested)) backend is required to run some of the experiments comparing our results with keras.


## **NN_lib:**
* **optimizer.py:** Contains the implementation of the different optimization algorithms (e.g. SGD, Momentum, Adam, Conjugate gradient, ..)
* **linesearches.py:** Contains the implementation of the backtracking and Armijo-Wolfe line searches
* **NN.py:** Contains the main functions to work create and use a neural network
* **layer.py:** Contains the functions concerning the creation and manipulation of a single layer of a neural network
* **activations.py:** Contains the implementation of a number of activation functions that can be used on the neural network' layers
* **regularization.py:** Contains the implementation of a number of regularization methods that can be used on the neural network' layers
* **loss_functions.py:** Contains the implementation of a number of loss functions that can be used for training a neural network
* **preproc.py:** Contains functions to define and manipulate a dataset object
* **validation.py:** Implements a number of methods such as cross validation and grid search for validating and exploring the parameters of a neural network model

Custom activations, regularizations and loss_functions may be defined, provided they implement the required interface.


## **External files and other folders:**
* **clean_keras_comp.py:**  Compares keras models with NN_lib models using the same parameters over randomly generated problems. Note that keras is required to run this file
* **monk_benchmark.py:** Compares keras with NN_lib on the [MONK problems](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems), a historical set of problems for comparing learning algorithms
* **digits_main.py:** Compares a keras model with a NN_lib one on a digit recognition task using the [MNIST digits dataset](http://yann.lecun.com/exdb/mnist/)
* **splitData.py:** Utility file for separating the training data for the ML cup dataset into two separate files for training and testing
* **grid_search_cup.py:** Contains the template to quickly validate a variety of models on the [ML cup challange](http://pages.di.unipi.it/micheli/DID/CUP-AA1/2017/data2017.html)
* **grid_optimizer.py:** Contains the template to quickly create, run and plot the behaviour of an optimization algorithms over a wide range of paramaters
* **test_functions.py:** Contains the implementation of a number of 2-variables test functions for testing optimization algorithms 
* **optimize_2d.py:** Contains the template to easily visually assess the behaviour of different optimization algorithms on `test_functions`. 
* **llss.py:** Contains the implementation of linear least squares solvers through the QR method and the normal equations one
* **llss_nn.py:** Contains the template for quickly comparing neural network, `llss` linear least squares solvers and `numpy`' linear least squares solvers on the ML cup dataset. The generlization error of the least squares solvers is calculated as well
* **conv_measures.py:** Contains the implementation of two rough measures of convergence speed and unstability given an array whose elements represent the value of a function over the iterations
* **ranker.py:** Displays a table or 2d plot visualization of rankings between the behaviour of different optimizers. Files with the history of the optimizers must be supplied. Such files are obtained from the execution of `grid_optimizer.py`
* **correlator.py:** Computes correlation matrices between the parameters of the optimizers and the results in terms of function value, unstability and convergence speed. Files with the history of the optimizers must be supplied. Such files are obtained from the execution of `grid_optimizer.py`
* **data:**: Contains the dataset that most of the experiments are based on
* **MONK_data:**: Contains the dataset of the MONK problems that some experiments are based on

## **Sample experiments**

### Comparison between NN_lib implementation and keras over a random dataset
Refer to `clean_keras_comp.py`
First we generate a random dataset as <br/>
`x = np.random.randn(500,inps)` <br/>
`y = np.random.randn(500,outs)`<br/>

Then we create a keras optimizer and one of our library as <br/>
`optimizer_keras = optims.Adamax(lr=clr,epsilon=1e-8)` <br/>
`optimizer = Adamax(lr=clr,b1=0.9,b2=0.999)`<br/>

Refer to `optimizers.py` for a comprehensive list of the available optimizers. Note that Adine and Conjugate Gradient are not available in keras.

To compare different network architectures we just need to modify the parameters of the added layers as<br/>
`NN.addLayer(inputs=inps,neurons=neurons,activation="tanh", rlambda=valr,regularization="EN", dropout=0.0,bias=0.0)` <br/>

And in keras as  <br/>
`model.add(Dense(neurons, activation= 'tanh' ,use_bias=True,input_dim=10, bias_initializer="zeros", kernel_regularizer=regularizers.l1_l2(valr[0],valr[1]),bias_regularizer=regularizers.l1_l2(valr[0],valr[1])))` <br/>

After running the file we obtain a plot with the function' value over the iterations both from keras optimizer as well as our own.


### Comparison between different combinations of parameters for the optimizers
Refer to `grid_optimizer`. The default version executes 3 trials of 100 iterations each for 48 configurations of the momentum optimizer. To test a different one simply provide the appropriate dictionary of parameters. Some examples are given in the file. Note that the constructor must be changed according to the optimizer. For instance for momentum it's: <br/>
`opt_list.append(Momentum(lr=param["lr"],eps=param["eps"],nesterov=param["nest"]))` <br/>

While for adam it's:<br/>
`opt_list.append(Adam(lr=param["lr"],b1=param["b1"],b2=param["b2"]))` <br/>

Parallelization is done over the trials.

The output produced is a pdf file with the 2d plots of the function value over the iterations and a history pkl object that may be used to compute additional information such as specified in `correlator.py` or `ranker.py`.

