// Constants for the DNN model on MNIST data

var modelName = 'DNN';
var dataset = 'MNIST';
var D = 784; // number of input features (28*28 pixels)


// Network parameters
var input_layer_bias = false;
var num_hidden_layers = 1;
var hidden_layer_num_units = 64;
var hidden_layer_relu = true;
var hidden_layer_bias = false;
var gradient_clip_bound = 4.0;
var logits_num_units = 10;
var logits_relu = false;
var logits_with_bias = false;



var batchSize = 500;  // at every iteration, the user will randomly pick a fixed size 'batchSize' of mini-batch samples from its dataset to compute the average gradient
var learnRate = .1;  // initial learning rate
var eps = .1; // privacy parameters, used to generate noise; smaller value means more stringent privacy requirement
var delta = .1; // privacy parameters for (\epsilon, \delta) privacy
var K = 10;   // number of classes 
var l = 0;	// l2 regularization paramete
var stepSize = 'const'; // indicates methods of calculating the learning rate/step size (constant, simple, sqrt, etc)
var noiseDist = 'Laplace'; // the noise distribution (laplacian distribution, gaussian distribution, etc)
var maxIter = 50000; // the maximum iteration the server can execute before stopping the optimization process (this is related to the privacy budget)
var randomize = 'True'; //If true, randomize the input data; otherwise use a fixed seed and non-randomized input.


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
exports.num_hidden_layers = num_hidden_layers;
exports.hidden_layer_num_units = hidden_layer_num_units;
exports.hidden_layer_relu = hidden_layer_relu;
exports.hidden_layer_bias = hidden_layer_bias;
exports.gradient_clip_bound = gradient_clip_bound;
exports.logits_num_units = logits_num_units;
exports.logits_relu = logits_relu;
exports.logits_with_bias = logits_with_bias;
exports.input_layer_bias = input_layer_bias;
