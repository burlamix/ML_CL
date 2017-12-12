import sys
from sklearn import preprocessing
from preproc import *
from NN import *
from optimizer import *
from validation import *
from keras.dataset import mnist


(x_train,y_train) ,(x_test, y_test) = mnist.load_data()