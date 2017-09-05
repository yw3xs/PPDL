#How to run the Server
1. Run the command:
```
$node MIDDLE-server.js arg1 arg2
```
  * ```arg1``` is the filename containing the model constants 
  * ```arg2``` is the path to python
  * Example: to run logistic regression model on my machine, the command is 
  
		```
		$node MIDDLE-server.js logReg.js /home/yang/anaconda2/bin/python	
		```
2. After the server is up, run the python users script (refer to the [README file](../CLIENT/README.md) in the CLIENT folder). 
3. The output of the server in the console and in the firebase are shown in the video clip

## files/subfolders in the SERVER folder
* MIDDLE-server.js: server file
* firebase.js: configuration for firebase
* model/: model folder which contains different model constants
* testset/: test set folder which contains test data set and parameter evaluation functions
