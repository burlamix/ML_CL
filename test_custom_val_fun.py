import validation
import preproc
from optimizer import *
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


acts=[["tanh","linear"]]#, ["sigmoid","linear"]]
opts=[optimizer10]#,optimizer6,optimizer7,optimizer8]
neurs=[[10,2]]
fgs = list()
trials = 1
for i in range(0,trials):
    fg,grid_res, pred = validation.grid_search(dataset, epochs=[2],batch_size=[1024],
                        n_layers=2, val_split=30,
                        activations=acts,cvfolds=1,verbose=2,loss_fun=["mse"],val_loss_fun="mse",
                     neurons=neurs ,optimizers=opts)   #with 10 neurons error! i don't now why
    fgs.append(fg)

print(grid_res.optimizer)
