import constants
import random
from random import gauss

def gendata():
    '''
    generate dataset
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

def gettotalerror(x,y,w1,w2):
    '''
    returns the total regressed error over the dataset x,y for the given params w1,w2
    :param x:
    :param y:
    :param w1:
    :param w2:
    :return:
    '''
    error=0
    for i in range(constants.NUMB_SAMPLES):
        error += (y[i] - w1*x[i]-w2)**2
    return error/len(x)
