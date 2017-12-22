from preproc import *
from NN import *
from keras.models import Sequential
import keras.optimizers as opts
from keras.layers import Dense
import keras
from keras import backend as k
import tensorflow as tf
np.set_printoptions(threshold=np.nan)

def bm_monk(optimizer, monk='monk1',act1='tanh',act2='sigmoid', reg=0.0, bs=32, epochs=1500,trials=5):
    #Setup -- don't mind this (just need to modify weights by hand)
    file = "bm_"+monk#+str(np.random.randn())
    s=monk + ",act1=" + act1 + ",weights=xavier(uniform),act2=" + act2 + ",reg=" + str(reg) + \
      ",bs=" + str(bs) + ",epochs=" + str(epochs) +","+optimizer.pprint()+"\n"
    h= open(file,"a")
    h.write(s)
    h.flush()
    mk = 1 if monk=='monk1' else (2 if monk=='monk2' else 3)
    units = 3 if monk=='monk1' or monk=='monk3' else 2

    '''
    x_train, y_train = load_monk("MONK_data/monks-"+str(mk)+".train")
    x_test, y_test = load_monk("MONK_data/monks-"+str(mk)+".test")
    #print("x_train-----------------",x_train)
    #print("y_train---------------",y_train)
    x_train = np.random.randn(50,17)
    y_train = np.random.randn(50,1)
    dataset = preproc.Dataset()
    dataset.init_train([x_train, y_train])
    dataset.init_test([x_test, y_test])
    #preproc.Preprocessor().shuffle(dataset)
    '''
    outs=1
    feat=5
    dataset = preproc.Dataset()
    x_train = (np.ones((1,feat)))*0
    y_train = np.random.randn(1,outs)
    print('yout',y_train)

    dataset.init_train((x_train,y_train))
    conv=0
    wgs = []
    units=10
    for i in range(0,trials):

        NN = NeuralNetwork()
        #NN.addLayer(2, 2, "sigmoid", weights=np.array([[0.15, 0.20], [0.25, 0.3]]), bias=0.35)
        #NN.addLayer(2, 2, "sigmoid", weights=np.array([[0.40, 0.45], [0.50, 0.55]]), bias=0.6)
        NN.addLayer(feat, outs, act1, rlambda=reg)
        #NN.addLayer(units, outs, act2, rlambda=reg)

        #Saving model weights--don't touch
        currwg = []
        for l in NN.layers:
            w=np.zeros((len(l.W[0])-1,len(l.W)))
            for i in range(0,len(l.W)):
                for j in range(1,len(l.W[i])):
                    w[j-1][i] = l.W[i][j]
            currwg.append(w)    #Actual weights
            currwg.append(np.zeros(len(l.W))) #Bias
        wgs.append(currwg)

        #for l in NN.layers:
            #print('W',l.W)
        (loss, acc, val_loss, val_acc, history) = \
            NN.fit_ds(dataset, epochs, optimizer, batch_size=bs, verbose=3)

        # Print model prediction on first 2 samples
        print('us',NN.predict(x_train[0:2]))
        #Save accuracies on file
        s = 'tr_acc='+str(acc)+', val_acc='+str(val_acc)+"\n"
        h.write(s)
        h.flush()
        conv+=1 if val_acc==1 else 0
    h.write("converged:"+str(conv)+"\nnot converged:"+str(trials-conv)+"\n")
    h.write("-------------------\n")


    #Now keras..
    conv = 0
    for i in range(0,trials):
        ini1 = keras.initializers.RandomNormal(mean=0.0, stddev=(2 / 20), seed=None)
        ini2 = keras.initializers.RandomNormal(mean=0.0, stddev=(2 / 4), seed=None)
        l1_reg=keras.regularizers.l2(0.0)

        model = Sequential()
        model.add(Dense(outs, activation=act1,
                        bias_initializer='zeros',
                        kernel_initializer="normal",
                        input_dim=feat, use_bias=True,kernel_regularizer=l1_reg))
        #model.add(Dense(outs, activation=act2, bias_initializer='zeros', kernel_initializer="normal", use_bias=True,kernel_regularizer=l1_reg))
        #model.add(Dense(7, activation='sigmoid',  kernel_initializer="normal", input_dim=2,use_bias=True,kernel_regularizer=l1_reg))
        #model.add(Dense(1, activation='sigmoid', kernel_initializer="normal",use_bias=True,kernel_regularizer=l1_reg))
        sgd = opts.SGD(lr=optimizer.getLr(), momentum=0.0, decay=0.00, nesterov=False)
        model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
        #Load weights
        wc=wgs[i]
       # wc[1]=[0.35,0.35]
       # wc[3]=[0.6,0.6]
        model.set_weights(wc)


        #print(wc[1])
        #dataset.train[1]=dataset.train[1].reshape((2,1))
        model.fit(dataset.train[0], y_train, batch_size=bs, verbose=0, epochs=epochs, shuffle=False)
        outputTensor = model.output
        listOfVariadatasetbleTensors = model.trainable_weights
        gradients = k.gradients(outputTensor, listOfVariadatasetbleTensors)

        # Print model prediction on first 2 samples
        print(model.predict(x_train[0:2]))

        #Save training and test accuracies
        acc = model.evaluate(x_train, y_train)[1]
        val_acc = model.evaluate(x_test, y_test)[1]
        s = 'tr_acc=' + str(acc) + ', val_acc=' + str(val_acc) + "\n"
        h.write(s)
        conv+=1 if val_acc==1 else 0
        h.flush()
    h.write("converged:"+str(conv)+"\nnot converged:"+str(trials-conv)+"\n")
    h.write("\n\n******************************************************************************************\n")

    h.close()