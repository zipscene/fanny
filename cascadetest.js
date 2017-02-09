var expect = require('chai').expect;
var fanny = require('./lib');

var createANN = fanny.createANN;
var createTrainingData = fanny.createTrainingData;
var XError = require('xerror');

// Inputs: A, B  Outputs: AND, OR, NAND, NOR, XOR
var booleanTrainingData = [
	[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ],
	[ [ 0, 1 ], [ 0, 1, 1, 0, 1 ] ],
	[ [ 0, 0 ], [ 0, 0, 1, 1, 0 ] ],
	[ [ 1, 1 ], [ 1, 1, 0, 0, 0 ] ]
];

function booleanThreshold(array) {
	return array.map(function(elem) {
		return (elem >= 0.5) ? 1 : 0;
	});
}


var ann = createANN({ layers: [ 2, 5 ], type: 'shortcut' });
ann.train(booleanTrainingData, { desiredError: 0, stopFunction: 'BIT', cascade: true }, 'default')
	.then(() => {
		expect(booleanThreshold(ann.run([ 1, 1 ]))).to.deep.equal([ 1, 1, 0, 0, 0 ]);
		expect(booleanThreshold(ann.run([ 1, 0 ]))).to.deep.equal([ 0, 1, 1, 0, 1 ]);
	});

