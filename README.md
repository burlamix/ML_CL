# titolo?	


## Structure of the library


The core of the structure is inside of NN. With *addLayer* it's possible to add new layers at the end of the neural network as you want, each layers with different attributes. 

## The core file of our project are inside de NN_lib folder

**NN_lib:**
* **optimizer:** Contains the algorithms of the different optimizers
* **linesearches:** Contains the main algorithms of line-search
* **NN:** In this file it's contain the main function for build, train, and use a neural network
* **activations:** Contains the possible activation fuction to use
* **layer:** Contains the function to menage the singular layer of the neural network
* **loss_functions:** Contains the possible loss function to use
* **regularization:** Contains the implemented regularizations
* **validation:** Contains the main function to make validation

### In the external file ca be found different settings and experiments
* **exsternal file:**
* **clean_keras_comp:**  a comparison between keras implementation and us implementation
* **digits_main:** test on the mnist, a well-know machine learning problem 
* **grid_optimizer:** experiment to see the behaviour of different parametrest of the optimizers
* **grid_search_cup:** main file use to do grid search on the ML cup
* **grid_search_cup_test:** --- come sopra, uno va levato 
* **llss:** contains the implemented linear least squares solver
* **llss_nn:** error comparison between neural network, our llss and llss off-the-shelf
* **monk_benchmark:** comparison keras and our neural-network
* **open_plk:** contains function to measures the convergence, and to open a formatted pks file
* **optimize_2d:** contains the main function and a experiment to plot 2D hystory of different optimizer
* **splitData:** contains the splitting settings, to divide training from test
* **test_functions:** contains the implemented 2D test function
* **to_line**: contains a comparison between our line search and the numpy line search