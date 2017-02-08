# FANNy: Modern Node.JS Bindings for FANN (Fast Artificial Neural Network library)

## Obligatory Brief Example

```js
var fanny = require('fanny');
// Create a neural network with 2 input nodes, 5 hidden nodes, and 1 output node
var ann = fanny.createANN({ layers: [ 2, 5, 1 ] });
// Boolean XOR function training dataset
var dataset = [
	{ input: [ 0, 0 ], output: [ 0 ] },
	{ input: [ 0, 1 ], output: [ 1 ] },
	{ input: [ 1, 0 ], output: [ 1 ] },
	{ input: [ 1, 1 ], output: [ 0 ] }
];
// Train until a MSE (mean squared error) of 0.025.  Returns a Promise.
ann.train(fanny.createTrainingData(dataset), { desiredError: 0.025 })
	.then(function() {
		// Training complete.  Do some test runs.
		// (exact output is different each time due to random weight initialization)
		console.log(ann.run([ 1, 0 ])); // [ 0.906... ]
		console.log(ann.run([ 1, 1 ])); // [ 0.132... ]
	});
```

## Supported Features

Nearly all features of FANN are supported, including different datatypes (float, double, fixed) and
training progress callbacks.  Multi-epoch training operations are asynchronous and run in a separate
thread.  `run()` can be called either synchronously or asynchronously, as its speed can vary widely
depending on the network size.

## Interfaces

FANNy's primary interface roughly mirrors FANN's C++ interface, but with a number of changes and
tweaks to make it better fit Javascript paradigms.  This is the interface described in the rest
of this file.

FANNy also provides a lower-level interface more directly in-line with FANN's own C++ interface.
This can be accessed by including the native addon itself: `require('fanny').getAddon('float')`.

## Getting FANN

This module is currently built on FANN git as of Jan. 2017.  However, the original author no longer seems to be
maintaining the library, and several bugs in the current official version break some features of
FANNy.  A maintained version with the appropriate bugfixes is available [here](https://github.com/crispy1989/fann).

When you npm install fanny, it should automatically fetch and compile this version of FANN
in a sandbox.  You will need standard build tools along with `cmake` installed.

## Creating a Neural Network

To create a new neural network, use the `createANN()` function.  It takes two parameters.
The first (required) parameter is an object containing configuration information for the
neural network.  The second parameter is an object containing a set of options.  See the
section on setting options for a list.

```js
var config = {
	type: 'standard', // 'standard' (default), 'sparse', or 'shortcut'
	layers: [ 2, 10, 3, 1 ], // Sizes of layers in the neural network.  Required.
	// connectionRate: 0.5, // Connection rate, for sparse networks
	datatype: 'float', // 'float' (default), 'double', or 'fixed', for libfloatfann, libdoublefann, libfixedfann, respectively
	activationFunctions: { // override default activation functions for layers or individual neurons
		// Possible values: 'LINEAR', 'THRESHOLD', 'THRESHOLD_SYMMETRIC', 'SIGMOID', 'SIGMOID_STEPWISE',
		// 'SIGMOID_SYMMETRIC', 'SIGMOID_SYMMETRIC_STEPWISE', 'GAUSSIAN', 'GAUSSIAN_SYMMETRIC',
		// 'ELLIOT', 'ELLIOT_SYMMETRIC', 'LINEAR_PIECE', 'LINEAR_PIECE_SYMMETRIC', 'SIN_SYMMETRIC',
		// 'COS_SYMMETRIC', 'SIN', 'COS'
		hidden: 'LINEAR', // all hidden nodes
		output: 'THRESHOLD', // all output nodes
		'2': 'LINEAR', // nodes on layer 2 (indexed from 0)
		'1-2': 'LINEAR' // node 2 on layer 1
	},
	activationSteepnesses: { // override default activation steepnesses for layers or individual neurons
		// keys are same as for activationFunctions
		hidden: 0.2
	}
};
var options = { ... };
var ann = fanny.createANN(config, options);
```

## Loading and Saving a Neural Network

Neural networks are saved by default in floating points.  FANN fixed point saving can be enabled by
passing boolean `true` as a second argument to `save()`.

`fanny.loadANN()` will, by default, load neural networks using libfloatfann.  A second argument can
be passed with the datatype ('float', 'double', or 'fixed') to change this.

```js
ann.save('/path/to/filename').then(...);
fanny.loadANN('/path/to/filename').then(function(ann) { ... });
```

## Options

Many of FANN's getter and setter functions are instead exposed as options that can easily
be set in batches.

```js
ann.setOption(name, value);
ann.getOption(name);
ann.setOptions({ name1: value1, name2: value2, ... });
ann.getOptions(); // returns object
```

The full list of possible options and available values can be found in `common-schema` format in
the file `src/ann.js`.

## Training Data

Training data is represented by a `TrainingData` class.  It's constructed, saved, and loaded like
an `ANN`:

```js
var trainingData = fanny.createTrainingData(data);
trainingData.save('/path/to/filename').then(...);
fanny.loadTrainingData('/path/to/filename').then(function(trainingData) { ... });
```

Note that training data must be instantiated with the same datatype as the ANN it's used with.
The functions `fanny.loadTrainingData()` and `fanny.createTrainingData()` both take an optional
second argument containing the datatype, if different from the default ('float').

The `data` parameter can take several different formats of data:

```js
var data1 = [
	{
		input: [ 0.2, 0.7, ... ],
		output: [ 0.3, 0.8, 0.5, ... ]
	},
	{
		input: [ ... ],
		output: [ ... ]
	},
	...
];
fanny.createTrainingData(data1);

var data2 = [
	[
		[ 0.2, 0.7, ... ],
		[ 0.3, 0.8, 0.5, ... ]
	],
	[
		[ ... ],
		[ ... ]
	],
	...
];
fanny.createTrainingData(data2);

var inputs = [
	[ 0.2, 0.7, ... ],
	...
];
var outputs = [
	[ 0.3, 0.8, 0.5, ... ],
	...
];
fanny.createTrainingData(inputs, outputs);
```

`TrainingData` also has several other methods that can get and manipulate the data.  These
are direct equivalents of their corresponding FANN functions.  Here are the available functions:

- `shuffle()`
- `merge()`
- `getLength()`
- `getNumInputs()`
- `getNumOutputs()`
- `getInputData()`
- `getOutputData()`
- `getOneInputData()`
- `getOneOutputData()`
- `getMinInput()`
- `getMaxInput()`
- `getMinOutput()`
- `getMaxOutput()`
- `scaleInput()`
- `scaleOutput()`
- `scale()`
- `subset()`
- `setData()`
- `clone()`

## Training

Training a single datapair is easy and synchronous:

```js
ann.trainOne([ <inputs>, <outputs> ]);
```

Training a single epoch with a training dataset returns a Promise:

```js
ann.train(trainingData).then(...);
```

To train for multiple epochs, you can pass a set of options to `train()` as a
second argument.  All options are optional, but it is highly recommended to set at least
`desiredError`.

```js
ann.train(trainingData, {
	desiredError: 0.05, // Stop training when this error (MSE by default) is reached
	maxEpochs: 1000, // Stop training if we do this many epochs
	stopFunction: 'MSE', // Determines the meaning of desiredError.  MSE is default.  'BIT' is for bitfail.
	cascade: false, // enable cascade training
	//maxNeurons: 100000, // Used instead of maxEpochs when cascade training
	progressInterval: 1 // Number of epochs between calling the progress function
}).then(...);
```

`train()` can also be given a third argument, a callback function that is called periodically
during training (defined by `progressInterval`).

```js
var progressFn = function(info) {
	console.log(info.iteration, info.mse, info.bitfail);
};
```

The progress function can optionally return `false` to cancel training (and immediately reject the promise).

Instead of passing a progress function as the third argument, the special value 'default' can be
passed (as a string) to enable FANN's default behavior of printing status reports to stdout.

## Running

The neural network can be run either synchronously or asynchronously:

```js
var outputs = ann.run(inputs);
ann.runAsync(inputs).then(function(outputs) { ... });
```

## Getting Current Information and Stats

The `ANN` object has a property called `info` containing current information about the network.  Keys include:

- `numInput`
- `numOutput`
- `totalNeurons`
- `totalConnections`
- `decimalPoint`
- `multiplier`
- `networkType`
- `connectionRate`
- `numLayers`

Each of these corresponds to a FANN getter.

## User Data

The `ANN` object has a property called `userData` which is initialized to an empty object.  You can store
any data you need to in there, and it will be saved and loaded with the neural network.  The data is
stored as a JSON object in the FANN `user_data_string` field.

## Other ANN Functions

These functions correspond directly (insofar as translation to Javascript allows) to functions
on the FANN C++ `neural_net` class.

- `randomizeWeights`
- `initWeights`
- `printConnections`
- `getMSE`
- `resetMSE`
- `printParameters`
- `getActivationFunction`
- `setActivationFunction`
- `setActivationFunctionLayer`
- `setActivationFunctionHidden`
- `setActivationFunctionOutput`
- `getActivationSteepness`
- `setActivationSteepness`
- `setActivationSteepnessLayer`
- `setActivationSteepnessHidden`
- `setActivationSteepnessOutput`
- `getLayerArray`
- `getBiasArray`
- `getConnectionArray`
- `setWeightArray`
- `setWeight`
- `scaleTrainingData`
- `descaleTrainingData`
- `setInputScalingParams`
- `setOutputScalingParams`
- `setScalingParams`
- `clearScalingParams`
- `scaleInput`
- `scaleOutput`
- `descaleInput`
- `descaleOutput`
- `testOne`
- `testData`


