import sys
from preproc import *
from NN_lib import loss_functions
from NN_lib.layer import Layer
import preproc

class NeuralNetwork:

    def __init__(self):
        #TODO initial layers
        self.layers = []

    def addLayer(self,inputs, neurons, activation, weights=np.array(None), bias=0,
                 regularization="L2", rlambda = 0.0,weights_init='fan_in',dropout=0.5):
        self.layers.append(Layer(inputs, neurons, activation, weights, bias,dropout=dropout,
                                 regularizer=regularization, rlambda=rlambda,weights_init=weights_init))

    def FP(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.getOutput(x)
        return x

    def BP(self, prediction, real, x_in):
        gradients = np.empty(len(self.layers),dtype=object)
        loss_func = self.loss_func.f(real,prediction)
        for i in range(len(self.layers)-1, -1, -1):

            logi = self.layers[i].activation.dxf(self.layers[i].currentOutput)
            logi = logi*self.layers[i].mask
            #print('curro:',np.count_nonzero(self.layers[i].currentOutput),
            #      's',np.array(self.layers[i].currentOutput).shape)
            #print('nonzeros:',np.count_nonzero(logi),'s',(logi).shape)
            #print('before:',np.count_nonzero(logi))

            #logi = logi*np.random.binomial(1,1-self.layers[i].dropout,self.layers[i].currentOutput.shape)/\
            #       ((1-self.layers[i].dropout))
            #print('zeris:',np.count_nonzero(logi))
            #print('after:',np.count_nonzero(logi),'s',logi.shape)
            if i==(len(self.layers)-1):
                e= self.loss_func.dxf(real, prediction)
                err = logi*e#self.loss_func[1](real, prediction)
                #print('errerewe',self.loss_func[1](prediction, real))
            else:
                err=logi*np.dot(err,self.layers[i+1].W[:,1:]) #error is derivative of activation
                #at current layer * (weights*error at next layer)
            if i==0:
                curro = x_in
            else:
                curro = self.layers[i-1].currentOutput
            #curro = curro*self.layers[i-1].mask if i>0 else curro

            curro = np.concatenate((np.ones((curro.shape[0], 1)), curro), axis=1)
            #curro *= np.random.binomial(1, self.layers[i].dropout, size=curro.shape)

            #print('err', curro)

            #print('curr',curro.shape)
            grad = np.dot(curro.transpose(),err)/(real.shape[0])
            #print(grad)

            #grad = (np.dot(curro.transpose(),err))
            self.layers[i].grad = grad
            #gradients.append(grad)
            gradients[-i]=grad
        return loss_func, gradients


    def regul(self):
        regul_loss = 0
        for l in self.layers:
            regul_loss+=l.regularize()
        #return regul_loss/len(self.input)      #TODO Input WTF?, pheraps
        return regul_loss

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
            self.set_weight(W)      #TODO TODO put this inside at FP and BP
            if only_fp:
                return self.loss_func.f(out_chunk,self.FP(in_chunk)) #+ self.regul()
            else:
                loss, grad = self.BP(self.FP(in_chunk), out_chunk, in_chunk)
                return loss + self.regul(),grad
        return g




    def fit(self, x_in, y_out, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0, val_set=None, val_loss_fun=None):
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
            sys.exit("Cannot use both a separate set and a split for validation ")

        # Check whether the user provided a properly formatted loss function
        self.loss_func = loss_functions.validate_loss(loss_func)

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

            #Randomly permute the data before each epoch
            perm = np.random.permutation(len(x_in))
            x_in = x_in[perm]
            y_out = y_out[perm]
            loss, acc = self.evaluate(x_in, y_out)

            for chunk in range(0, len(x_in), batch_size):
                cap = min([len(x_in), chunk + batch_size])

                update = optimizer.optimize(self.f(x_in[chunk:cap], y_out[chunk:cap]), self.get_weight())

                #predicted = self.FP(x_in[chunk:cap],)
                for j in range(0, len(self.layers)):
                    #print("-----1----",update[j])
                    #print("-----2----",(self.reguldx(j) / batch_size).transpose())
                    update[j]-= (self.reguldx(j) / 1).transpose()
                    update[j]/=1

                self.set_weight(update)

                #for j in range(0, len(self.layers)):
                    #self.layers[j].W = self.layers[j].W + update[j].transpose() - (self.reguldx(j) / batch_size)
                    #self.layers[j].W = self.layers[j].W + update[-j - 1].transpose() - (self.reguldx(j) / batch_size)


            val_loss = None
            val_acc = None
            if (val_split > 0 or val_set!=None ):
                val_loss, val_acc = self.evaluate(validation_x, validation_y,val_loss_fun)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            history['tr_loss'].append(loss)
            history['tr_acc'].append(acc)


            if(verbose >= 2):
                #loss,acc = self.evaluate(x_in,y_out)
                #print("loss=",loss,"        acc=",acc)
                print (i,' loss = {0:.8f} '.format(loss),'accuracy = {0:.8f} '.format(acc),
                       (' val_loss = {0:.8f} '.format(val_loss),
                       ' val_acc = {0:.8f} '.format(val_acc))if (val_split>0 or val_set!=None) else "")
            #TODO inefficente..
        #TODO proper output formatting
            #if(verbose>=2 and (val_split>0 or val_set!=None) ):
              #  print("Validation loss:"+str(val_loss)+' val acc:'+str(val_acc))
        if (verbose >= 1 and (val_split > 0 or val_set != None)):
            print("Validation loss:" + str(val_loss) + ' val acc:' + str(val_acc))
        return (loss, acc, val_loss, val_acc, history)

    def fit_ds(self, dataset, epochs, optimizer, batch_size=-1, loss_func="mse", val_split=0, verbose=0,val_set=None,val_loss_fun=None):
        return self.fit(dataset.train[0], dataset.train[1], epochs, optimizer, batch_size, loss_func, val_split, verbose,val_set,val_loss_fun)


    def evaluate(self,x_in,y_out, loss_fun=None):

        if loss_fun == None:loss_fun = self.loss_func
        loss_fun = loss_functions.validate_loss(loss_fun)

        real = self.FP(x_in)

        #val_loss_func = self.loss_func[0](real,dataset.test[0]) #+ self.regul()        TODO TODO TODO MUST cambiare così
        val_loss_func = loss_fun.f(real,y_out)# + self.regul()
        
        correct=0
        errate=0
        accuracy=0
        #TODO -> make it work in the general case
        for i in range(0,real.shape[0]):
            #if ( (real[i][0]>0.5 and y_out[i][0]==1) or (real[i][0]<=0.5 and y_out[i][0]==0) ): #TODO time? make it automaticaly
            if ( (real[i][0]>0 and y_out[i][0]==1) or (real[i][0]<=0 and y_out[i][0]==-1) ): #TODO time? make it automaticaly
                correct = correct +1
            else:
                errate = errate + 1

        accuracy = correct/real.size #TOCHECK this is the accuracy that whant micheli?
        
        #if(accuracy>0.999):exit(1)
        return val_loss_func, accuracy


    def predict(self, x_in):
        return self.FP(x_in)

    def initialize_random_weight(self, method='fan_in'):
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
        W = np.empty(len(self.layers), dtype=object)
        for i in range(0,len(self.layers)):
            W[i] = self.layers[i].W.transpose()
        return W



  #  def w_update(update):
   #     for i in range (0,len(self.layers)):
    #        self.layers[i].W = self.layers[i].W+update[-i-1].transpose()
