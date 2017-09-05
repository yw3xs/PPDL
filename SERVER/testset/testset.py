from __future__ import division
import sys
import numpy as np
import tensorflow as tf

def loadTest(file, output):
    '''
    This function read the test set and convert it to tensors 
    file: original MNIST test set in the DATA/MNIST folder (10000 samples)    
    rows: the number of rows of the output testset (<= 10000)
    output: the resulting csv dataset with original label column removed, the new 0/1 label is in the last column    
    '''
    X = np.genfromtxt(file, delimiter=',', max_rows = np.int(rows))    
    newlabel = (X[:, 0] == np.int(base)).astype(int)
    X = np.c_[np.delete(X, 0, 1), newlabel]
    np.savetxt(output, X, delimiter = ',')

if __name__ == '__main__':
    loadTest(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
