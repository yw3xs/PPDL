// Server for MIDDLE 
// Author: Yang Wang
// Set up the server for the MIDDLE prototype using Firebase real-time database
// Second version (DNN + MNIST data set)
// argv[2]: the model file
// argv[3]: the path to python

// Integrate the Firebase Realtime Database SDKs and include the constants for the model 
var firebase = require('firebase');
var constants = require('./model/' + process.argv[2]);  
//var constants = require('./model/DNN.js');  // for testing purpose
var config = require('./firebase.js');
var fs = require('fs');
var util = require('util');
var test_log = fs.createWriteStream('./testset/test.log', {flags : 'w'});
var param_log = fs.createWriteStream('./param.log', {flags : 'w'});

// Set up the configuration for the Firebase database. 
firebase.initializeApp(config.config);

// Get a reference to the Firebase database service
var db = firebase.database().ref();

// Set up different child databases to store data/parameters for the model fitting process.
// 'parameter' contains the model parameters we are trying to estimate and its iteration number.
// 'model' contains the constants of the model, which are checked out by the clients at the beginning of the fitting process, and updated whenever there is a change. 
// 'users' contains information uploaded by the clients/users (gradients, iteration number, etc)
// 'final' holds the final parameter after completing the optimization process
var parameter = db.child('parameter'); 
var model = db.child('model'); 
var users = db.child('users');
var final = db.child('final');

// Clear up the leftover from previous run
parameter.remove();
model.remove(); 
users.remove();
final.remove();

// Set up the model 
model.update({
	modelName: constants.modelName,
	dataset: constants.dataset,
	D: constants.D,
	batchSize: constants.batchSize,
	rate: constants.learnRate,
	eps: constants.eps,
	K: constants.K,
	l: constants.l,
	baseValue: constants.baseValue,
	stepSize: constants.stepSize,
	noiseDist: constants.noiseDist, 	
	maxIter: constants.maxIter, 
	threshold: constants.threshold, 
	tolerance: constants.tolerance, 
	num_hidden_layers: constants.num_hidden_layers,
	hidden_layer_num_units: constants.hidden_layer_num_units,
	hidden_layer_relu: constants.hidden_layer_relu,
	hidden_layer_bias: constants.hidden_layer_bias,
	hidden_layer_relu: constants.hidden_layer_relu,
	gradient_clip_bound: constants.gradient_clip_bound,
	logits_num_units: constants.logits_num_units,
	logits_relu: constants.logits_relu,
	logits_with_bias: constants.logits_with_bias,
	input_layer_bias: constants.input_layer_bias
})
console.log('Model set up!');

// the iteration number starts from 0
var initIter = 0; 
// Prepare the test set using python and initialize the parameter database 


var param_len = 0;
// number of neurons and bias node at each layer
var neuron_bias = [constants.D + constants.input_layer_bias];
for (i = 0; i < constants.hidden_layer_num_units; i++){
	neuron_bias.push(constants.hidden_layer_num_units[i] + constants.hidden_layer_bias[i]);
}
neuron_bias.push(constants.K + constants.logits_with_bias);

// TODO hard-coded parameters, change it later!
var Param1 = new Array((constants.D + constants.input_layer_bias) * constants.hidden_layer_num_units);
for(var i=0; i< Param1.length; i++) {
    Param1[i] = (Math.random() - 0.5) / 10;
}

var Param2 = new Array((constants.hidden_layer_num_units + constants.hidden_layer_bias) * constants.logits_num_units);
for(var i=0; i< Param2.length; i++) {
    Param2[i] = (Math.random() - 0.5) / 10;
}



//var Param1 = new Array((constants.D + constants.input_layer_bias) * constants.hidden_layer_num_units[0] + 1).join('0').split('').map(parseFloat);
//var Param2 = new Array((constants.hidden_layer_num_units[0] + constants.hidden_layer_bias[0]) * constants.hidden_layer_num_units[1] + 1).join('0').split('').map(parseFloat);
//var Param3 = new Array((constants.hidden_layer_num_units[1] + constants.hidden_layer_bias[1]) * constants.logits_num_units + 1).join('0').split('').map(parseFloat);

var param_num = [Param1.length, Param2.length];

console.log("Dimension of first layer parameters: " + Param1.length);
console.log("Dimension of second layer parameters: " + Param2.length);

model.update({
	param_num: param_num
})

parameter.update({
		parameter1: Param1,
		parameter2: Param2,
		iteration: initIter, 
});	

////////////////////////////////// all tested above ///////////////
var PythonShell = require('python-shell');
var options = {
 	mode: 'text',
  	pythonPath: process.argv[3],  // path to python in the working machine
//  pythonPath: '/home/yang/anaconda2/bin/python',  // for testing purpose
  	pythonOptions: ['-u'],
  	scriptPath: './testset'
  	//args: pythonArgs
};
/*
// use the python script to evaluate the model

	// run python shell inside node.js
/*
if (constants.modelName == 'DNN' && constants.dataset == 'MNIST'){
	pythonArgs = ['../DATA/MNIST/MNIST_test.csv', constants.baseValue, constants.testrows, './testset/MNIST_test_bin.csv']
}

var options = {
 	mode: 'text',
  	pythonPath: process.argv[3],  // path to python in the working machine
//  pythonPath: '/home/yang/anaconda2/bin/python',  // for testing purpose
  	pythonOptions: ['-u'],
  	scriptPath: './testset',
  	args: pythonArgs
};
PythonShell.run('testset.py', options, function (err, results) {
  	if (err) console.log(err);
  	console.log('Test set ready!');
	// initiatlize/update the parameter database	
  	parameter.update({
		parameter1: Param1,
		parameter2: Param2,
		parameter3: Param3,
		iteration: initIter, 
	});	
});
*/
var currentParam1;
var currentParam2;
var currentIter;

// read current parameter from the data base; run the callback function every time there is a update to the parameters.
parameter.on('value', function(snapshot) {
	if (snapshot.exists()) {
		var serverParam = snapshot.val();
		currentParam1 = serverParam.parameter1;
		//onsole.log(currentParam1);
		currentParam2 = serverParam.parameter2;
		currentIter = serverParam.iteration;		
		console.log('Iteration ' + currentIter);
		//param_log.write('Parameter: [' + currentParam + ']\n');
		//console.log('===================\n');
		/*
		if (currentIter > constants.maxIter) {
			// check whether the maximum iteration is achieved or not	
			console.log('Maximum iteration achieved!')	
			final.update({
				finalParam: currentParam
			});
			parameter.off();
			users.off();
		} else {
		*/						
			// evaluate the parameter using the test set, execute the computation in python
		var test = new PythonShell('./testset/testIt.py');	

		test.send(JSON.stringify(currentParam1));	// pass the current parameters to the python script						
		test.send(JSON.stringify(currentParam2));

		test.on('message', function (message) {	 
			console.log("test start!");  	
			//console.log(message);		
			accuracy = Math.round(Number(message)*10000)/100		    	
	    	console.log('The accuracy is ' + accuracy + '%!');	
	    	console.log("test end!");  
	    	/*		    	
	    	if (message < constants.tolerance){
	    		console.log('Satisfactory error rate achieved!');
				final.update({
					finalParam: currentParam
				});	
				parameter.off();	
				users.off();
	    	}	
	    	*/
		});	
		test.end(function (err) {
			if (err){
				console.log("error here!");
				throw err;
			}			
		});									
		
	}
});

// When new users come in
users.on('child_added', function(snapshot) {
	console.log('New user: ' + snapshot.val().userID);	
	updateParam(snapshot, currentParam1, currentParam2, currentIter);		
});

// When existing users update their gradient
users.on('child_changed', function(snapshot) {
	console.log('Changed user: ' + snapshot.val().userID);	
	updateParam(snapshot, currentParam1, currentParam2, currentIter);		
});

// sum up all the gradients of the samples in the batch
function updateParam(snapshot, oldParam1, oldParam2, oldIter) {
	var user = snapshot.val();
	var grad1 = user.grad1;
	var grad2 = user.grad2;
	var userID = user.userID;
	var userIter = user.userIter;
	//if (userIter == oldIter){			
		console.log('Parameter update #' + oldIter + ' starts! (User: ' + userID + ')');
		var newParam1 = [];
		var newParam2 = [];
		for (i = 0; i < oldParam1.length; i++) { 
				newParam1[i] = oldParam1[i] + grad1[i];
		}
		for (i = 0; i < oldParam2.length; i++) { 
				newParam2[i] = oldParam2[i] + grad2[i];
		}
		console.log('Parameter update #' + oldIter + ' ends!');
		parameter.update({			
			parameter1: newParam1,
			parameter2: newParam2,
			iteration: oldIter + 1	
		});
		/*
		if(constants.stepSize=='const'){
			learningRate = constants.learnRate;
			for (i = 0; i < length; i++) { 
				newParam[i] = oldParam[i] - (learningRate * grad[i]);
			}
		}
		else if(constants.stepSize=='simple'){
			learningRate = constants.learnRate/(currentIter+1);
			for (i = 0; i < length; i++) { 
				newParam[i] = oldParam[i] - (learningRate * grad[i]);
			}
		}
		else if(constants.stepSize=='sqrt'){
			learningRate = constants.learnRate/Math.sqrt((currentIter+1));
			for (i = 0; i < length; i++) { 
				newParam[i] = oldParam[i] - (learningRate * grad[i]);
			}
		}
		console.log('Parameter update #' + oldIter + ' ends!');
		parameter.update({			
			parameters: newParam,
			iteration: oldIter + 1,	
		});
		*/
		
	//}
	return true;
}


