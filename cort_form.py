import pickle
import preproc
from NN_lib import NN
import numpy as np
import preproc
from NN_lib import validation
from NN_lib.NN import *
from NN_lib.optimizers import *
import matplotlib.pyplot as plt
from NN_lib import regularizations
from matplotlib.backends.backend_pdf import PdfPages
import pickle

np.random.seed(5)


train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"
dataset = preproc.Dataset()
dataset.init_train(preproc.load_data(path=train_data_path, target=True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(path=test_data_path, target=False, header_l=10))

preprocessor = preproc.Preprocessor()

preprocessor.shuffle(dataset)

adamax1 = Adamax(lr=0.016,b1=0.9,b2=0.999)


nn = NN.NeuralNetwork()


NN = NeuralNetwork()
NN.addLayer(inputs=10,neurons=50,activation="sigmoid", rlambda=(0.0001,0.0001),regularization="EN",
            dropout=0)
NN.addLayer(inputs=50,neurons=2,activation="linear",rlambda=(0.0001,0.0001),regularization="EN",
            dropout=0)


NN.fit_ds( dataset,epochs=3105, optimizer=adamax1 ,batch_size=dataset.train[0].shape[0],verbose=2,loss_func="mee")


with open('final_model.pkl', 'wb') as output:
    pickle.dump(NN, output, pickle.HIGHEST_PROTOCOL)




with open('final_model.pkl', 'rb') as output: 
	nn = pickle.load(output)

out = nn.predict(dataset.test[0])

b = np.ones((out.shape[0],out.shape[1]+1))
b[:,1:]=out
b[:,0]=np.arange(0,out.shape[0])+1

with open('final_pred.csv', 'w') as output: 
	for l in b:
		output.write(str(int(l[0]))+",{0:.6f}".format(l[1])+",{0:.6f}".format(l[2])+"\n")


np.savetxt('test.out', out, delimiter=',')
exit(1)
np.random.seed(321)
out = np.random.randn(10,2)
b = np.ones((out.shape[0],out.shape[1]+1))
b[:,1:]=out
b[:,0]=np.arange(0,out.shape[0])
print(b)