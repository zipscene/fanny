var expect = require('chai').expect;
var fanny = require('../lib');

var createANN = fanny.createANN;
var loadANN = fanny.loadANN;
var createTrainingData = fanny.createTrainingData;
var loadTrainingData = fanny.loadTrainingData;

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

describe('Tests', function() {

	it('basic test', function() {
		var ann = createANN({ layers: [ 2, 20, 5 ] });
		return ann.train(booleanTrainingData, { desiredError: 0.01 })
			.then(() => {
				expect(booleanThreshold(ann.run([ 1, 1 ]))).to.deep.equal([ 1, 1, 0, 0, 0 ]);
				expect(booleanThreshold(ann.run([ 1, 0 ]))).to.deep.equal([ 0, 1, 1, 0, 1 ]);
			});
	});

});
