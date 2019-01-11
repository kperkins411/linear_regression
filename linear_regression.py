#generate data with linear component
import random
from random import gauss

NUMB_SAMPLES = 50
RAND_MAX_VAL =10
RAND_MIN_VAL =0
MAX_RISE = 10
NUM_EPOCHS =10
LR=0.0021

def gendata():
    x = [val for val in range(NUMB_SAMPLES)]
    y=[random.random()*(RAND_MAX_VAL-RAND_MIN_VAL)+RAND_MIN_VAL + val + MAX_RISE + RAND_MAX_VAL*gauss(0,1) for val in range(NUMB_SAMPLES)]
    # y = [random.random() * (RAND_MAX_VAL - RAND_MIN_VAL) + RAND_MIN_VAL + MAX_RISE for val in range(NUMB_SAMPLES)]
    # y=[9 for _ in range(NUMB_SAMPLES)]
    return(x,y)

import matplotlib.pyplot as plt
def setlimits(x,y):
    '''
    make sure plot is fixed so graph moves
    :param x:
    :param y:
    :return:
    '''
    plt.xlim(min(x), max(x))
    plt.ylim(min(y),max(y))
    # plt.xlim(0, NUMB_SAMPLES)
    # plt.ylim(0, NUMB_SAMPLES)

    # Don't mess with the limits!
    plt.autoscale(False)

def plotdata(x,y, w1, w2):
    plt.clf()
    setlimits(x,y)
    #here is the linear regressed line
    l = [w1*n+w2 for n in x]

    plt.scatter(x, y)   #the random data
    plt.plot(x,l,'-r')  #the line
    plt.pause(0.1)

def gen_weights():
    #make this a little more random?
    return .5,10

def gettotalerror(x,y,w1,w2):
    error=0
    for i in range(NUMB_SAMPLES):
        error += (y[i] - w1*x[i]-w2)**2
    return error/len(x)

def main():
    w1,w2 = gen_weights()   #create weights
    x,y=gendata()           #create random data

    plt.ion()   #interactive on


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plotdata(x, y, w1, w2)

    cnt = 0
    for _ in range(NUM_EPOCHS):

        #calc the tot gradient
        sum_w1 = 0
        sum_w2 = 0
        for i in range(NUMB_SAMPLES):
            sum_w1 += -(y[i] - w1*x[i] -w2)*x[i]
            sum_w2 += -(y[i] - w1*x[i] -w2)

        #divide by numb_samples
        grad_w1 = sum_w1/NUMB_SAMPLES
        grad_w2 = sum_w2/NUMB_SAMPLES

        # adjust w1 and w2
        w1 = w1 - LR*grad_w1
        w2 = w2 - LR*grad_w2

        print(f"w1={w1}  w2={w2} totalerror={gettotalerror(x,y,w1,w2)}")


        if(cnt%25==0):
            plotdata(x,y,w1,w2)
        cnt+=1

    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()

