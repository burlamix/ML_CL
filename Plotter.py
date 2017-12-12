from matplotlib import pyplot

def loss_over_epochs(history):
    loss = history['tr_loss']
    print(len(loss))
    pyplot.plot(loss)
    pyplot.show()