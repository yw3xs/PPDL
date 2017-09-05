# How to load MNIST datasets
1. Download the following original dataset files from [the MNIST database](http://yann.lecun.com/exdb/mnist/ "THE MNIST DATABASE of handwritten digits") into the folder "original_dataset".
  * train-images-idx3-ubyte:  training set images  
  * train-labels-idx1-ubyte:  training set labels 
  * t10k-images-idx3-ubyte:   test set images 
  * t10k-labels-idx1-ubyte:   test set labels 
2. Run the python script
```shell
$./loadMNIST.py
```
3. Two csv files `MNIST_train.csv` and `MNIST_test.csv` are generated.


