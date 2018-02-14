from NN_lib.optimizers import *
from NN_lib import loss_functions
from NN_lib.NN import *
from NN_lib import preproc
from NN_lib import linesearches
import llss
import numpy as np


def normeqs_solve(X,y):
    #Solve the least squares problem with our implementation of normal equations
    normeqs_sol = llss.norm_eqs(X,y,l=regul) #Get the optimal weights
    normeqs_pred = np.dot(X,normeqs_sol) #Predict the output with the optimal weights
    normeqs_loss = loss_functions.mse(y,normeqs_pred) #Calculate the loss
    return normeqs_sol, normeqs_pred, normeqs_loss

def QR_solve(X,y):
    #Solve the least squares problem with our implementation of QR
    qr_sol = llss.QR_solver(X,y,l=regul) #Get the optimal weights
    qr_pred = np.dot(X,qr_sol) #Predict the output with the optimal weights
    qr_loss = loss_functions.mse(y,qr_pred) #Calculate the loss
    return qr_sol, qr_pred, qr_loss

np.random.seed(5)
regul = 0.0


#Load and shuffle dataset
dataset = preproc.Dataset()
train_data_path = "data/myTrain.csv"
test_data_path = "data/myTest.csv"
dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=0, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=True, header_l=0, targets=2))

preprocessor = preproc.Preprocessor()
preprocessor.shuffle(dataset)


#Define line searches
amg = linesearches.ArmijoWolfe(m1=1e-1, m2=0.1, lr=1, min_lr=1e-11, scale_r=0.95, max_iter=2000)
bt = linesearches.BackTracking(lr=1, m1=1e-4, scale_r=0.4, min_lr=1e-11, max_iter=20000)

epochs = 500
optimizer1 = ConjugateGradient( lr=0.1, ls=amg, beta_f="PR", restart=-1)
optimizer2 = Adam(lr=0.3)

#Train with a linear neural network - 1 layer with linear activation.
NN = NeuralNetwork()
NN.addLayer(inputs=10,neurons=2,activation="linear",rlambda=regul,regularization="L2")
(loss, acc, val_loss, val_acc, history2)=\
    NN.fit_ds( dataset,epochs, optimizer2  ,val_split=0,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mse")

#Prepare the data for the linear least square solver - include the bias
biasedX = dataset.train[0]
biasedX = np.concatenate((np.ones((biasedX.shape[0],1)),biasedX),axis=1)

#Solve the least squares problem with numpy' linear algebra library
numpy_sol = np.linalg.lstsq(biasedX,dataset.train[1]) #Get the optimal weights
numpy_pred = np.dot(biasedX,numpy_sol[0]) #Predict the output with the optimal weights
numpy_loss = loss_functions.mse(dataset.train[1],numpy_pred) #Calculate the loss

normeqs_sol, _, normeqs_loss = normeqs_solve(biasedX, dataset.train[1])
qr_sol, _, qr_loss = QR_solve(biasedX, dataset.train[1])

print('\nNeural network error:',loss)
print('Numpy linear least squares solver error:',numpy_loss)
print('Normal equations error:',normeqs_loss)
print('QR solver error:',qr_loss)

######## GENERALIZATION ERROR WITH LLSSs ############
#Prepare test data for the linear least square solver - include the bias
biasedXtest = dataset.test[0]
biasedXtest = np.concatenate((np.ones((biasedXtest.shape[0],1)),biasedXtest),axis=1)

qr_pred = np.dot(biasedXtest, qr_sol)  # Predict the output with the optimal weights
qr_loss = loss_functions.mse(dataset.test[1], qr_pred)  # Calculate the loss
normeqs_pred = np.dot(biasedXtest, normeqs_sol)  # Predict the output with the optimal weights
normeqs_loss = loss_functions.mse(dataset.test[1], qr_pred)  # Calculate the loss

print("\n*** GENERALIZATION ERROR ***\n")
print('QR solver error on test set:',qr_loss)
print('Normal equations error on test set:',normeqs_loss)
