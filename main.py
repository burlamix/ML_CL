import sys
from sklearn import preprocessing
from preproc import *
from NN import *
from optimizer import *
from validation import *
train_data_path = "data/ML-CUP17-TR.csv"
test_data_path = "data/ML-CUP17-TS.csv"


np.random.seed(5)
dataset = Dataset()
dataset.init_train(load_data(train_data_path, True, header_l=10, targets=2))
dataset.init_test(load_data(test_data_path, False, header_l=10))

preprocessor = Preprocessor()


#preprocessing




        #gradient calculation is the same for all optimizers,
        #the update rule changes for each one
        #while backprop set layer.gradient
     #   self.optimizer.optimize(self.loss_func(dataset.train))


#Real dataset
activation = Activation(sigmoid,sigmoddxf,sigmoid)
activation1 = Activation(linear,lineardxf,sigmoid)
NN = NeuralNetwork()#276828657
NN.addLayer(10,20,activation1)
NN.addLayer(20,2,activation1)
optimizer = SimpleOptimizer(0.05)
preprocessor.normalize(dataset,norm_output=False)



NN.fit(dataset, 100, optimizer, batch_size=1016)

a = NN.evaluate(dataset)
fg,grid_res, pred = grid_search(dataset, epochs=[10], n_layers=2,
                       neurons=[[2,2],[55,2],[20,2],[25,2],[30,2]] ,optimizers=[optimizer])   #with 10 neurons error! i don't now why

#grid_res.fit(dataset,100,1016)
print(grid_res.neurons)
b = grid_res.NN.evaluate(dataset)
print(a)#result normal fit
print(b)#result from grid search

for i in fg:
    print(i['val_loss'])

print("PRED:"+str(len(pred)))

#Toy dataset
toyx = np.asarray([[0.05,0.1]]) #TODO make it work with arrays
NN = NeuralNetwork()
NN.addLayer(2,2,activation,weights=np.array([[0.15,0.20],[0.25,0.3]]),bias=0.35)
NN.addLayer(2,2,activation,weights=np.array([[0.40,0.45],[0.50,0.55]]),bias=0.6)
dataset.train[0] = toyx
dataset.train[1] = np.array([0.01, 0.99])
#NN.fit(dataset, 1111, optimizer)

#Toy dataset
np.random.seed(5)
optimizer = SimpleOptimizer(lr=0.1)
dataset.train[0] = np.random.rand(500,10)
dataset.train[1] = np.random.rand(500,1)
NN = NeuralNetwork()
#NN.addLayer(2,1,activation1,weights=np.array([[0.6, 0.8]]), bias=0.2)
NN.addLayer(10,20,activation1)
NN.addLayer(20,1,activation1)
#preprocessor.normalize(dataset)
#NN.fit(dataset, 8000, optimizer,batch_size=500)
#s=0
#for l in NN.layers:
#    s+=np.sum(np.abs(l.W))
#print(np.sum(s))
#print(NN.FP(x_in=dataset.train[0]))




'''o = NN.FP(toyx)
print("==========================")
o1 = NN.BP(o,np.array([0.01,0.99]),toyx)
for l in NN.layers:
    l.W = l.W - 0.5*l.grad.transpose()
    print(l.W)
print(o1)
NN = NeuralNetwork()
NN.addLayer(type,regularizer,neurons,activation..)

NN.addLayers(type,[array of neurons], <number of layers>,
                                [array of regularizers],[array of activations])
NN.compile(loss_func, optimizer,(metric)..)

NN.fit(dataset, batch_size, epochs, cvfolds=0, vsplit=0) #only one of cvfolds, vsplit
#grid search function
NN.test()'''

#TODO min max when min=max -> if min=max then normalize as el/(length of list)
#TODO make dataset work when input is array and not matrix
#TODO gradient checking http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
#TODO compare with keras setting seed
