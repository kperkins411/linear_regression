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

    plt.pause(0.1)


def main(x,y,w1,w2,lr):
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
        w1, w2 = backprop(lr, w1, w2, x, y)
        tot_error = utils.gettotalerror(x,y,w1,w2)
        print(f"w1={w1}  w2={w2} totalerror={tot_error}")

        if(cnt%5==0):
            plotdata(x,y,w1,w2, lr, tot_error,epoch_num)
        cnt+=1

    plt.ioff()
    plt.show()


LR_EPOCHS = 100
MAX_LR = .001
MIN_LR = .000001
FEW_TEST_CYCLES=4   #sum the errors for this many cycles to see if they diverge
def find_learning_rate(x,y,w1_init,w2_init,lr_epochs = LR_EPOCHS,max_lr=MAX_LR, min_lr=MIN_LR,
                       test_cycles=FEW_TEST_CYCLES):
    '''
    creates a list of learning rates from min_lr to max_lr. Then checks each for LR_EPOCHS to calculate totalError
    the lr with the smallest total error is the one that will converge the fastest
    :param x: data,x
    :param y: data,y
    :param w1_init:
    :param w2_init:
    :param lr_epochs:
    :param max_lr:
    :param min_lr:
    :param test_cycles:
    :return:
    '''
    LR= np.linspace(min_lr, max_lr, lr_epochs).tolist() #evenly spaced list of lr's
    totalerrors=[]
    for lr in LR:
        w1, w2 = w1_init, w2_init  # create weights
        tmp_errors=[]
        for _ in range(test_cycles):
            w1, w2 = backprop(lr, w1, w2, x, y)
            tmp_errors.append(utils.gettotalerror(x,y,w1,w2))

        # totalerrors.append(min(tmp_errors))
        totalerrors.append(tmp_errors[test_cycles-1])

        print(f"w1={w1}  w2={w2} lr={lr} totalerror={utils.gettotalerror(x,y,w1,w2)}")

    #here is the smallest error
    smallest_index = totalerrors.index(min(totalerrors))
    best_lr = LR[smallest_index]
    print(f"best learning rate={best_lr}")

    #enable below to see plot
    plt.title("Learning rate finder ")
    plt.xlabel("error")
    plt.ylabel("learning rate")
    plt.ylim(min(totalerrors)-30,min(totalerrors)+30)  #lowest learning rate sets highest error (to display)
    plt.xlim(min(LR), max(LR))
    plt.autoscale(False)
    plt.plot(LR, totalerrors, '-r')  # the line

    plt.annotate(
        'The minimum error',
        xy=(best_lr, min(totalerrors)), arrowprops=dict(arrowstyle='->'), xytext=(best_lr, min(totalerrors)-5))
    plt.show()

    return best_lr


def backprop(lr, w1, w2, x, y):
    '''
    backprop over errorfunction to adjust w1 and w2
    :param lr: learning rate
    :param w1: weight1
    :param w2: weight2
    :param x: dependent
    :param y: independent
    :return:
    '''
    # calc the tot gradient
    sum_w1 = 0
    sum_w2 = 0
    for i in range(constants.NUMB_SAMPLES):
        #sum de/dw1 and de/dw2
        sum_w1 += -(y[i] - w1 * x[i] - w2) * x[i]
        sum_w2 += -(y[i] - w1 * x[i] - w2)

    # divide by numb_samples
    grad_w1 = sum_w1 / constants.NUMB_SAMPLES
    grad_w2 = sum_w2 / constants.NUMB_SAMPLES

    # adjust w1 and w2
    w1 = w1 - lr * grad_w1
    w2 = w2 - lr * grad_w2
    return w1, w2


if __name__=="__main__":
    random.seed(1)  #repeatability

    x, y = utils.gendata()  # create random data
    w1, w2 = constants.INITIAL_WEIGHTS  # initial weights

    #first find the largest learning rate that allows model convergence
    best_lr = find_learning_rate(x,y,w1,w2)

    #run the regressor
    main(x,y,w1,w2,lr=best_lr)
