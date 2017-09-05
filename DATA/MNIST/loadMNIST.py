#!/usr/bin/env python
# Load the MNIST database of handwritten digits
# Before running the scripts, download the original data files from http://yann.lecun.com/exdb/mnist/ into the folder /original_dataset. There are 4 files:

# train-images-idx3-ubyte: training set images 
# train-labels-idx1-ubyte: training set labels 
# t10k-images-idx3-ubyte: test set images 
# t10k-labels-idx1-ubyte: test set labels

# The training set contains 60000 examples, and the test set 10000 examples. Each image is 28 pixels by 28 pixels. Pixels are organized row-wise. Pixel values are from 0 to 1. 0 means background (white), 1 means foreground (black). The file format is described in the reference: http://yann.lecun.com/exdb/mnist/

from __future__ import division

def loadMNIST(image, label, out, n):
    '''
    This function merges the pixels values with the labels and converts the original data files into csv files.
    image: image file
    label: label file
    out: output file
    n: number of samples
    '''
    i = open(image, 'rb')    # open the image file in binary mode
    l = open(label, 'rb')    # open the label file in binary mode
    o = open(out, 'w')

    i.read(16)    # first 16 bytes of the image file are metadata
    l.read(8)    # first 8 bytes of the label file are metadata
    images = []

    for k in xrange(n):
        image = [ord(l.read(1))]    # this is the label
        image.append(1)     # this is the intercept term
        # the images were centered in a 28x28 image
        for j in xrange(28*28):  
            image.append(round(ord(i.read(1))/255, 3))    # convert the pixel value to [0,1] and append the pixel of the image        
        images.append(image)

    for image in images:
        o.write(','.join(str(pixel) for pixel in image)+'\n')
        
    i.close()
    l.close()
    o.close()
    
def main():
    loadMNIST('./original_dataset/train-images-idx3-ubyte', './original_dataset/train-labels-idx1-ubyte', 'MNIST_train.csv', 60000)
    loadMNIST('./original_dataset/t10k-images-idx3-ubyte', './original_dataset/t10k-labels-idx1-ubyte', 'MNIST_test.csv', 10000)

if __name__ == '__main__':
    main()