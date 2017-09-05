from __future__ import division
import numpy as np


def binLabel(X, position, value = 0):
    '''
    This function takes the ndarray X and convert it to 2-class label dataset 
    X: orignal dataset
    position: the column number of the label
    value: when the original label is equal to value, it is set to 1; otherwise 0
    output: the resulting ndarray dataset with original label column removed, the new label is in the last column
    '''
    N = X.shape[0]
    newlabel = (X[:, position] == value).astype(int)
    return np.c_[np.delete(X, position, 1), newlabel]
    
         
def getGradient(param, sample, l):
    '''
    This function takes the batch-size sample and calculate the average gradient
    param: the current parameter (list)
    sample: the batch-size sample (ndarray)
    l: lambda for regularized objective function
    '''
    N,p = sample.shape
    X_sample = sample[:, :(p-1)]
    Y_sample = sample[:, p-1]
    param = np.asarray(param)
    
    prob = np.exp(np.dot(X_sample, param)) / (1 + np.exp(np.dot(X_sample, param)))
    
    g = (np.dot(np.transpose(X_sample), (prob - Y_sample)))/N - l * param

    return g

    


