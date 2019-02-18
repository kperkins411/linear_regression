from scipy.stats import logistic

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
from scipy.special import expit


def sigmoid_array(x):
   return 1 / (1 + np.exp(-x))

def predict(features, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1

  from https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#id11
  '''
  z = np.dot(features, weights)
  return sigmoid_array(z)


def decision_boundary(probs):
    '''
    convert preds to 1's and 0's
    :param probs:
    :return:
    '''
    func = lambda t: 1 if t>.5 else 0
    vfunc= np.vectorize(func)
    return vfunc(probs)

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

        tot_error = get_error(w, x, y)

        if(cnt%5==0):
            plt.clf()

            correctly_classified = y-decision_boundary(predict(x,w))

            f = lambda x: 'g' if x == 0 else 'r'
            vf = np.vectorize(f)
            correctly_classified = vf(correctly_classified)

            plt.scatter(x[:,0], x[:,1], c=correctly_classified)

            plt.pause(0.1)

        if(cnt%40 == 0):
            # see if learning rate has changed
            lr = utils.find_learning_rate_vectorize(x, y, w, backprop_vectorize,get_error, plt_lrs=False)
        cnt+=1

    plt.ioff()
    plt.show()


def get_error(w, x, y):
    # error
    preds = predict(x,w)
    error = -y*np.log(preds) - (1-y)*np.log(1-preds)
    error = error.sum()/len(x[:,0])
    print(f"Error is {error}")
    return error


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
    numb_samples=len(x[:,0])

    preds = predict(x,w)
    # tmp = (y-expit( w @ x.T))
    # preds = np.expand_dims(preds, axis=1)

    # sum derivatives
    tots = preds-y
    tots = np.expand_dims(tots, axis=1)
    grads = x*tots

    #add and divide by numb_samples
    total = np.sum(grads, axis=0)
    grad_w = total / numb_samples

    # adjust w1 and w2
    w = w -lr * grad_w

    return w

if __name__=="__main__":
    np.random.seed(1)  #repeatability

    x, y = utils.genClustereddData()  # create random data

    # plt.scatter(x[:,0], x[:,1], c=y)
    # plt.show()

    #initialize the weights
    w = np.random.randn(3)
    w[2]=1.0

    # first find the largest learning rate that allows model convergence
    best_lr = utils.find_learning_rate_vectorize(x, y, w, backprop_vectorize,get_error, plt_lrs=False)

    # best_lr = .08
    #run the regressor
    main_vectorize(x, y, w, lr=best_lr)
