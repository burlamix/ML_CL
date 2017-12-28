import preproc
from NN_lib.optimizers import *
from NN_lib import activations, validation
import time

train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"
np.random.seed(11)
dataset = preproc.Dataset()

dataset.init_train(preproc.load_data(train_data_path, True, header_l=10, targets=2))
dataset.init_test(preproc.load_data(test_data_path, False, header_l=10))

optimizer2 = SimpleOptimizer(lr=0.2)
optimizer3 = SimpleOptimizer(lr=0.5)
optimizer4 = SimpleOptimizer(lr=0.7)
optimizer5 = Momentum(lr=0.5,eps=0.5)
optimizer6 = Momentum(lr=0.5,eps=0.9)
optimizer7 = Momentum(lr=0.3,eps=0.5)
optimizer8 = Momentum(lr=0.3,eps=0.9)
optimizer9 = Adam(lr=0.005,b1=0.9,b2=0.999)
optimizer10 = RMSProp(lr=0.0051)

rel = activations.Activation(activations.relu, activations.reludx)
acts=[["relu","linear"]]#, ["sigmoid","linear"]]
opts=[optimizer10]#,optimizer6,optimizer7,optimizer8]
neurs=[[10,2]]
fgs = list()
trials = 1

start = time.time()
#for i in range(0,3000):
    #loss_functions.mdx(np.random.randn(100,2),np.random.randn(100,2))
#    lineardxf(np.random.randn(100,3))
#end = time.time()
#print(end-start)
#exit(1)
#dataset.train[0] = np.random.randn(1024,10)
#dataset.train[1] = np.random.randn(1024,2)

#preproc.Preprocessor().normalize(dataset,norm_output=False)
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[100], batch_size=[1024],
                            n_layers=2, neurons=neurs, val_split=30, cvfolds=1,
                            activations=acts, optimizers=opts, loss_fun=["mse"],
                            val_loss_fun="mse", verbose=2) 
    fgs.append(fg)

#print(grid_res.NN.evaluate(dataset.train[0],dataset.train[1],loss_fun="mse"))
