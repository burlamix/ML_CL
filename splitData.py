import numpy as np
from NN_lib.optimizers import *
from NN_lib.NN import *
from NN_lib import preproc

np.random.seed(15)
dataset = preproc.Dataset()

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"

#Load data
dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))

preprocessor = preproc.Preprocessor()

#Shuffle according to seed
preprocessor.shuffle(dataset)

#Split it into 75% training and 25% test
(dataset.train[0], dataset.test[0]), (dataset.train[1], dataset.test[1]) = \
    dataset.split_train_percent(75)


#Move training on its own file
i=1
with open("myTrain.csv", "w") as f:
    for sample in range(0,len(dataset.train[0])):
        f.write(str(i))
        for value in range(0,len(dataset.train[0][sample])):
           f.write(','+str(dataset.train[0][sample][value]))
        for value in range(0,len(dataset.train[1][sample])):
            f.write(',' + str(dataset.train[1][sample][value]))
        i+=1
        f.write('\n')

#Move test on its own file
with open("myTest.csv", "w") as f:
    for sample in range(0,len(dataset.test[0])):
        f.write(str(i))
        for value in range(0,len(dataset.test[0][sample])):
           f.write(','+str(dataset.test[0][sample][value]))
        for value in range(0,len(dataset.test[1][sample])):
            f.write(',' + str(dataset.test[1][sample][value]))
        i+=1
        f.write('\n')
