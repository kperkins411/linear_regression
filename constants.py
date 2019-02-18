'''
constants for regression projects
'''
NUMB_SAMPLES = 100
RAND_MAX_VAL =10
RAND_MIN_VAL =0
MAX_RISE = 20
NUM_EPOCHS =10000

INITIAL_WEIGHTS= [.5,0]

#for learning rate finder
LR_EPOCHS = 100
MAX_LR = 1
MIN_LR = .000001
FEW_TEST_CYCLES=4   #sum the errors for this many cycles to see if they diverge