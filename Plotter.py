from matplotlib import pyplot

def loss_over_epochs(history):



    loss = history['tr_loss']
    pyplot.plot(loss,label="training")


    loss = history['val_loss']
    pyplot.plot(loss,label="validation")


    tr_acc = history['tr_acc']
    pyplot.plot(tr_acc,label="training acc")


    val_acc = history['val_acc']
    pyplot.plot(val_acc,label="validation acc")


    pyplot.legend(loc='best')

    pyplot.show()