var fanny = require('./index');
// Create a neural network with 2 input nodes, 5 hidden nodes, and 1 output node
var ann = fanny.createANN({ layers: [ 2, 5, 1 ] });
// Boolean XOR function training dataset
var dataset = [
	{ input: [ 0, 0 ], output: [ 0 ] },
	{ input: [ 0, 1 ], output: [ 1 ] },
	{ input: [ 1, 0 ], output: [ 1 ] },
	{ input: [ 1, 1 ], output: [ 0 ] }
];
// Train until a MSE (mean squared error) of 0.05.  Returns a Promise.
ann.train(fanny.createTrainingData(dataset), { desiredError: 0.025 })
	.then(function() {
		// Training complete.  Do some test runs.
		// (exact output is different each time due to random weight initialization)
		console.log(ann.run([ 1, 0 ])); // [ 0.906... ]
		console.log(ann.run([ 1, 1 ])); // [ 0.132... ]
	});

