import numpy as np
import sys
from preproc import *
from loss_functions import losses
import types
from layer import Layer
import preproc

class NeuralNetwork:

    def __init__(self):
        #TODO initial layers
        self.layers = []

    def addLayer(self,inputs, neurons, activation, weights=np.array(None), bias=0,
                 regularization="L2", rlambda = 0.0):
        self.layers.append(Layer(inputs, neurons, activation, weights, bias,
                                 regularization, rlambda))

    def FP(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.getOutput(x)
        return x

    def BP(self, prediction, real, x_in):
        gradients = []
        loss_func = self.loss_func[0](real,prediction) #+ self.regul()
        for i in range(len(self.layers)-1, -1, -1):

            logi = self.layers[i].activation.dxf(self.layers[i].currentOutput)

            if i==(len(self.layers)-1):
                err = logi*self.loss_func[1](prediction, real)
            else:
                err=np.dot(err,self.layers[i+1].W[:,1:])*logi #error is derivative of activation
                #at current layer * (weights*error at next layer)
            if i==0:
                curro = x_in
            else:
                curro = self.layers[i-1].currentOutput
            curro = np.concatenate((np.ones((curro.shape[0], 1)), curro), axis=1)
            grad = (np.dot(curro.transpose(),err)/(real.shape[0]))
            self.layers[i].grad = grad
            gradients.append(grad)
        return loss_func, np.array(gradients)


    '''def regul(self):
        regul_loss = 0
        for l in self.layers:
            regul_loss+=l.regularize()
        return regul_loss/len(self.input)
'''
#    def reguldx(self):
#        regul_loss = 0
#        for l in self.layers:
#            regul_loss+=l.regularizedx()
#        return regul_loss/len(self.dataset.train[0])

    def reguldx(self,i):
        return self.layers[i].regularizedx()


    def f(self, in_chunk, out_chunk):
        def g(W):
            return self.BP(self.FP(in_chunk), out_chunk, in_chunk)
        return g

    def fit(self, x_in, y_out, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0):
        # ***GENERAL DESCRIPTION***
        # loss_func: can either be a string refering to a standardized defined
        # loss functions or a tuple where the first element is a loss function
        # and the second element is the corresponding derivative
        #
        # batch_size: the dimension of the samples to use for each update step.
        # note that a higher value leads to higher stability and parallelization
        # capabilities, possibly at the cost of a higher number of updates
        ######################################################################

        # Check whether the user provided a properly formatted loss function
        if isinstance(loss_func[0], types.FunctionType) and \
                isinstance(loss_func[1], types.FunctionType):
            self.loss_func = loss_func
        else:
            # Otherwise check whether a the specified loss function exists
            try:
                self.loss_func = losses[loss_func]
            except KeyError:
                sys.exit("Loss function undefined")

        if batch_size < 0 or batch_size > (len(x_in)):  # TODO more check
            batch_size = len(x_in)

        if (val_split>0):
            (x_in, validation_x),(y_out, validation_y) = \
                preproc.split_percent(x_in, y_out, val_split)
        for i in range(0, epochs):
            #print(i)
            for chunk in range(0, len(x_in), batch_size):
                cap = min([len(x_in), chunk + batch_size])

                update = optimizer.optimize(self.f(x_in[chunk:cap], y_out[chunk:cap]), "ciao")

                #predicted = self.FP(x_in[chunk:cap],)

                for j in range(0, len(self.layers)):
                    self.layers[j].W = self.layers[j].W + update[-j - 1].transpose() - (self.reguldx(j) / batch_size)

            if(verbose >= 1):
                loss,acc = self.evaluate(x_in,y_out)
                #print("loss=",loss,"        acc=",acc)
                print (i,' loss = {0:.8f} '.format(loss),'accuracy = {0:.8f} '.format(acc))
            #TODO inefficente..

        val_loss = None
        tr_loss = self.evaluate(x_in, y_out)
        if(val_split>0):val_loss = self.evaluate(validation_x, validation_y)
        if(verbose>=2):print("Training loss:"+str(self.loss_func[0](self.FP(x_in), y_out)))
        #TODO proper output formatting
        if(verbose>=1 and val_split>0):
            print("Validation loss:"+str(val_loss))
        return (tr_loss,val_loss)

    def fit_ds(self, dataset, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0):
        return self.fit(dataset.train[0], dataset.train[1], epochs, optimizer, batch_size, loss_func, val_split, verbose)

    ''' #TODO this needs to be on test set not train
    #TODO Also not very handy to use a dataset here..
    def evaluate(self,dataset):

        real = self.FP(dataset.train[0])

        #val_loss_func = self.loss_func[0](real,dataset.test[0]) #+ self.regul()        TODO TODO TODO MUST cambiare così
        val_loss_func = self.loss_func[0](real,dataset.train[1]) #+ self.regul()
        return val_loss_func
  '''
    def evaluate(self,x_in,y_out):

        real = self.FP(x_in)

        #val_loss_func = self.loss_func[0](real,dataset.test[0]) #+ self.regul()        TODO TODO TODO MUST cambiare così
        val_loss_func = self.loss_func[0](real,y_out) #+ self.regul()
        
        correct=0
        errate=0
        accuracy=0
       # print(real)
        #print(y_out)

        for i in range(0,real.shape[0]):
            if ( (real[i][0]>0.5 and y_out[i][0]==1) or (real[i][0]<=0.5 and y_out[i][0]==0) ): 
                correct = correct +1
            else:
                errate = errate + 1

        accuracy = correct/real.size #TOCHECK this is the accuracy that whant micheli?


        return val_loss_func, accuracy


    def predict(self, x_in):
        return self.FP(x_in)

    def initialize_random_weight(self):
    # inizialize all weight of all layers
    #                                       Optional set how random inizialize??
        for layer in self.layers:
            layer.initialize_random_weights()

  #  def w_update(update):
   #     for i in range (0,len(self.layers)):
    #        self.layers[i].W = self.layers[i].W+update[-i-1].transpose()
