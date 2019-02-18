'''
generates a bunch of random data
calculates the largest learning rate to use
then tries to fit the following line to that data
y=w1*x + w2 to it

lots of cheesy plotting
'''
import random
import constants
import utils
import numpy as np
import matplotlib.pyplot as plt

def gettotalerror_vectorize(w,x,y):
    '''
    returns the total regressed error over the dataset x,y for the given params w1,w2
    :param x:
    :param y:
    :param w
    :return:
    '''
    return np.sum((1 / 2) * (y - w @ x) ** 2, axis=1) / len(x[0])
def main_vectorize(x,y,w,lr):
    '''
    dynamically plot linear regression
    :param x:
    :param y:
    :param w1:
    :param w2:
    :param lr: learning rate, calculate from LR_finder
    :return:
    '''
    plt.ion()   #interactive on
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cnt = 0
    for epoch_num in range(constants.NUM_EPOCHS):
        w = backprop_vectorize(lr, w, x, y)
        tot_error = np.sum((1/2)*(y-w@x)**2, axis=1)/len(x[0])

        if(cnt%5==0):
            utils.plotdata(x[0,:],y,w[0,0],w[0,1], lr, tot_error[0],epoch_num)
            # print(f"w1={w[0,0]}  w2={w[0,1]} totalerror={tot_error}")

        # if(cnt%40 == 0):
        #     # see if learning rate has changed
        #     lr = utils.find_learning_rate_vectorize(x, y, w, backprop_vectorize,gettotalerror_vectorize  plt_lrs = False)
        cnt+=1

    plt.ioff()
    plt.show()


def backprop_vectorize(lr, w, x, y):
    '''
    backprop over errorfunction to adjust w1 and w2
    :param lr: learning rate
    :param w: weight vector (same size as x, add 1 to end if necessary)
    :param x: values vector dependent
    :param y: results vector independent
    :return:
    '''
    #the first param has all the values, get the number of columns for the number of samples
    numb_samples=len(x[0])

    # sum derivatives
    total = np.sum(-(y - w @ x) * x, axis=1)

    # divide by numb_samples
    grad_w = total / numb_samples

    # adjust w1 and w2
    w = w - np.transpose(lr * grad_w)

    return w

if __name__=="__main__":
    np.random.seed(1)  #repeatability

    x, y = utils.gendata()  # create random data
    w1, w2 = constants.INITIAL_WEIGHTS  # initial weights

    w = np.array([[w1, w2]])
    x = np.asarray(x)
    x = np.vstack((x, np.ones((x.shape[0]), dtype=x.dtype)))

    # first find the largest learning rate that allows model convergence
    best_lr = utils.find_learning_rate_vectorize(x, y, w, backprop_vectorize,gettotalerror_vectorize, plt_lrs=False)

    #run the regressor
    main_vectorize(x, y, w, lr=best_lr)
