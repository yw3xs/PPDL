# Functions for data set preparation
from __future__ import division
import numpy as np
import csv
    
## Load data for the user (1000 samples per user)
def loadData(files, rows):
    '''
    This function loads dataset for the user
    files: original MNIST trainig set
    rows: the row index of the subset owned by the user
    '''
    with open(files, 'r') as fin:
         reader=csv.reader(fin)
         result=np.array([[float(s) for s in row] for i,row in enumerate(reader) if i in rows])
    return result

# sample minibatch
def sampleData(X, userSize, batchSize):
    '''
    This function samples 'batchSize' data points from the training set owned by the user
    '''
    # Randomly choose (batchSize) samples
    ind = np.random.choice(range(userSize),size=batchSize,replace=False)        
    return (X[ind,:])
