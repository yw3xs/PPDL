// Constants for the logistic regression model on MNIST data

var modelName = 'logReg';
var dataset = 'MNIST';
var D = 785; // number of features (28*28 pixels + intercept)
var batchSize = 100;  // at every iteration, the user will randomly pick a fixed size 'batchSize' of mini-batch samples from its dataset to compute the average gradient
var learnRate = .1;  // initial learning rate
var eps = .1; // privacy parameters, used to generate noise; smaller value means more stringent privacy requirement
var K = 2;   // number of classes 
var l = 0;	// l2 regularization parameter
var stepSize = 'const'; // indicates methods of calculating the learning rate/step size (constant, simple, sqrt, etc)
var noiseDist = 'Laplace'; // the noise distribution (laplacian distribution, gaussian distribution, etc)
var maxIter = 100; // the maximum iteration the server can execute before stopping the optimization process (this is related to the privacy budget)
var tolerance = .05; // the threshold for the error on the test set, if lower than this threshold, stop the iteration and output the parameters to the database 'final'
var threshold = .5; // the probability threshold for logistic regression. if the predicted probability greater than this value, label the sample as 1; otherwise 0.
var baseValue = 5; // used to convert the original 10-class MNIST dataset to a binary label data set. 
var testrows = 3000; // the sample size of the test set.

if(stepSize != 'const' && stepSize != 'sqrt' && stepSize != 'simple')
{
	throw new Error('Invalid descent Algorithm');
}

exports.modelName = modelName;
exports.dataset = dataset;
exports.D = D;
exports.batchSize = batchSize;
exports.learnRate = learnRate;
exports.eps = eps;
exports.K = K;
exports.l = l;
exports.stepSize = stepSize;
exports.noiseDist = noiseDist;
exports.maxIter = maxIter;
exports.tolerance = tolerance;
exports.threshold = threshold;
exports.baseValue = baseValue;
exports.testrows = testrows;