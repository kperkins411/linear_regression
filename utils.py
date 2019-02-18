import constants
import random
import numpy as np
from random import gauss
import matplotlib.pyplot as plt

def genClustereddData():
    '''
    generate dataset for logistic regression
    :return: x,y dataset
    '''

    # Generate three random clusters of 2D data bbb
    NSAMPLES = 30
    NDIMS = 2
    NCLUSTERS = 2

    center_cluster_A = [1,1]
    center_cluster_B = [1.5,1.5]

    #define the clusters
    A = np.random.randn(NSAMPLES, NDIMS) + center_cluster_A
    B = np.random.randn(NSAMPLES, NDIMS) + center_cluster_B

    x=np.vstack((A,B))
    q= np.ones((1,len(x[:,0])))
    x = np.hstack((x, q.T))
    y = np.hstack((np.zeros(A.shape[0]), np.ones(B.shape[0])))
    return (x, y)


def gendata():
    '''
    generate dataset for linear regression
    :return: x,y dataset
    '''
    x = [val for val in range(constants.NUMB_SAMPLES)]
    y=[random.random()*(constants.RAND_MAX_VAL-constants.RAND_MIN_VAL)+constants.RAND_MIN_VAL + val + constants.MAX_RISE + constants.RAND_MAX_VAL*gauss(0,1) for val in range(constants.NUMB_SAMPLES)]
    # y = [random.random() * (RAND_MAX_VAL - RAND_MIN_VAL) + RAND_MIN_VAL + MAX_RISE for val in range(NUMB_SAMPLES)]
    # y=[9 for _ in range(constants.NUMB_SAMPLES)]
    return(x,y)

def gen_weights(initial_val = 0.5,num_weights=2):
    '''
    generate list of weights initialized to initial_val
    :param num_weights:
    :return: list of weights
    '''
    #make this a little more random?
    return[initial_val for _ in range(num_weights)]



def _setplotlimits(x,y):
    '''
    make sure plot scale is fixed so line moves
    :param x:
    :param y:
    :return:
    '''
    plt.xlim(min(x), max(x))
    plt.ylim(min(y),max(y))

    # Don't mess with the limits!
    plt.autoscale(False)

def plotdata(x,y, w1, w2, lr, tot_error,epoch_numb):
    '''
    dynamicly plots data with a bit of a pause
    '''
    plt.clf()
    _setplotlimits(x,y)
    #here is the linear regressed line
    l = [w1*n+w2 for n in x]

    plt.legend

    plt.scatter(x, y)   #the random data
    plt.plot(x,l,'-r')  #the line

    #have it begin 5% in from left and 15% down from top  "final w1={:.7}".format(w1)
    plt.text(.05*constants.NUMB_SAMPLES, .85*max(y), "Learning rate ={:.7}".format(lr)+ "\n"
                                                     "Total error   ={:.7}".format(tot_error)+ "\n"
                                                     "Epoch number  ={:}".format(epoch_numb))

    plt.pause(0.01)

def find_learning_rate_vectorize(x,y,w_init,backprop_func,get_error_func, lr_epochs = constants.LR_EPOCHS,max_lr=constants.MAX_LR,
                       min_lr=constants.MIN_LR, test_cycles=constants.FEW_TEST_CYCLES, plt_lrs = True):
    '''
    creates a list of learning rates from min_lr to max_lr. Then checks each for LR_EPOCHS to calculate totalError
    the lr with the smallest total error is the one that will converge the fastest
    :param x: data,x
    :param y: data,y
    :param w
    :param backprop_func function that calculates gradient for params
    :param get_error_func function that calculates the error
    :param lr_epochs:
    :param max_lr:
    :param min_lr:
    :param test_cycles:
    :return:
    '''
    LR= np.linspace(min_lr, max_lr, lr_epochs).tolist() #evenly spaced list of lr's
    totalerrors=[]
    for lr in LR:
        w=w_init  # create weights

        #train for a few epochs to see what kind of error we get
        #if converging, then last epoch should have lowest error for this lr
        for _ in range(test_cycles):
            w = backprop_func(lr, w, x, y)

        err = get_error_func(w,x,y)
        totalerrors.append(err)
        # print(f"w1={w[0,0]}  w2={w[0,1]} lr={lr} totalerror={err}")

    #here is the smallest error
    smallest_index = totalerrors.index(min(totalerrors))
    best_lr = LR[smallest_index]
    print(f"best learning rate={best_lr}")

    #enable below to see plot
    if(plt_lrs==True):
        plt.title("Learning rate finder ")
        plt.ylabel("error")
        plt.xlabel("learning rate")
        plt.ylim(min(totalerrors)-30,min(totalerrors)+30)  #lowest learning rate sets highest error (to display)
        plt.xlim(min(LR), max(LR))
        plt.autoscale(False)
        plt.plot(LR, totalerrors, '-r')  # the line

        plt.annotate(
            'The minimum error',
            xy=(best_lr, min(totalerrors)), arrowprops=dict(arrowstyle='->'), xytext=(best_lr, min(totalerrors)-5))
        plt.show()

    return best_lr
