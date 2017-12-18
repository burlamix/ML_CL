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
            #grad = (np.dot(curro.transpose(),err))
            self.layers[i].grad = grad
            #gradients.append(grad)
            gradients.insert(0,grad)
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


#    def f(self, in_chunk, out_chunk):
#        def g(W):
#            return self.BP(self.FP(in_chunk), out_chunk, in_chunk)
#        return g

    def f(self, in_chunk, out_chunk):
        def g(W,only_fp=False):

            self.set_weight(W)
            if only_fp:
                return self.FP(in_chunk)
            else:
                return self.BP(self.FP(in_chunk), out_chunk, in_chunk)
        return g




    def fit(self, x_in, y_out, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0, val_set=None):
        # ***GENERAL DESCRIPTION***
        # loss_func: can either be a string refering to a standardized defined
        # loss functions or a tuple where the first element is a loss function
        # and the second element is the corresponding derivative
        #
        # batch_size: the dimension of the samples to use for each update step.
        # note that a higher value leads to higher stability and parallelization
        # capabilities, possibly at the cost of a higher number of updates
        ######################################################################

        if(val_set!=None and val_split>0 ):
            sys.exit(" o uno o l'altro OH ")

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
        elif val_set!=None:
            validation_x=val_set[0]
            validation_y=val_set[1]


        history = {'tr_loss':[], 'val_loss':[], 'tr_acc':[], 'val_acc':[]}
        for i in range(0, epochs):
            #print(i)
            for chunk in range(0, len(x_in), batch_size):
                cap = min([len(x_in), chunk + batch_size])

                update = optimizer.optimize(self.f(x_in[chunk:cap], y_out[chunk:cap]), self.get_weight())

                #predicted = self.FP(x_in[chunk:cap],)
                self.set_weight(update)
                for j in range(0, len(self.layers)):
                    self.layers[j].W -= (self.reguldx(j) / batch_size)
                #for j in range(0, len(self.layers)):
                    #self.layers[j].W = self.layers[j].W + update[j].transpose() - (self.reguldx(j) / batch_size)
                    #self.layers[j].W = self.layers[j].W + update[-j - 1].transpose() - (self.reguldx(j) / batch_size)

            loss, acc = self.evaluate(x_in, y_out)
            val_loss = None
            val_acc = None
            if (val_split > 0 or val_set!=None ):
                val_loss, val_acc = self.evaluate(validation_x, validation_y)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            history['tr_loss'].append(loss)
            history['tr_acc'].append(acc)


            if(verbose >= 2):
                #loss,acc = self.evaluate(x_in,y_out)
                #print("loss=",loss,"        acc=",acc)
                print (i,' loss = {0:.8f} '.format(loss),'accuracy = {0:.8f} '.format(acc))
            #TODO inefficente..
        #TODO proper output formatting
        if(verbose>=1 and val_split>0 or val_set!=None ):
            print("Validation loss:"+str(val_loss)+' val acc:'+str(val_acc))
        return (loss, acc, val_loss, val_acc, history)

    def fit_ds(self, dataset, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0,val_set=None):
            return self.fit(dataset.train[0], dataset.train[1], epochs, optimizer, batch_size, loss_func, val_split, verbose,val_set)


    def evaluate(self,x_in,y_out):

        real = self.FP(x_in)

        #val_loss_func = self.loss_func[0](real,dataset.test[0]) #+ self.regul()        TODO TODO TODO MUST cambiare così
        val_loss_func = self.loss_func[0](real,y_out) #+ self.regul()
        
        correct=0
        errate=0
        accuracy=0
        #TODO -> make it work in the general case
        for i in range(0,real.shape[0]):
            if ( (real[i][0]>0.5 and y_out[i][0]==1) or (real[i][0]<=0.5 and y_out[i][0]==0) ): #TODO time? make it automaticaly
            #if ( (real[i][0]>0 and y_out[i][0]==1) or (real[i][0]<=0 and y_out[i][0]==-1) ): #TODO time? make it automaticaly
                correct = correct +1
            else:
                errate = errate + 1

        accuracy = correct/real.size #TOCHECK this is the accuracy that whant micheli?
        
        #if(accuracy>0.999):exit(1)
        return val_loss_func, accuracy


    def predict(self, x_in):
        return self.FP(x_in)

    def initialize_random_weight(self, method='xavier'):
        for layer in self.layers:
            layer.initialize_random_weights(method)

    #take a list of matrix, for inizializa layers wheight
    def set_weight(self, W):
        for i in range(len(W)-1,-1,-1):
            #self.layers[i].set_weights(W[i].transpose())
            self.layers[i].set_weights(W[i])
        #for layer,W_i in zip(self.layers,W):
        #    layer.set_weights(W_i)

    def get_weight(self):
        W=[]

        for layer in self.layers:
            #W.insert(0,layer.W)
            W.append(layer.W.transpose())
        return np.array(W)



  #  def w_update(update):
   #     for i in range (0,len(self.layers)):
    #        self.layers[i].W = self.layers[i].W+update[-i-1].transpose()
