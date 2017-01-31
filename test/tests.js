var expect = require('chai').expect;
var fanny = require('../lib');

var annOptionsSchema = fanny.annOptionsSchema;
var createANN = fanny.createANN;
var loadANN = fanny.loadANN;
var createTrainingData = fanny.createTrainingData;
var loadTrainingData = fanny.loadTrainingData;
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

describe('Tests', function() {

	it('basic test', function() {
		var ann = createANN({ layers: [ 2, 20, 5 ] });
		return ann.train(booleanTrainingData, { desiredError: 0.01 })
			.then(() => {
				expect(booleanThreshold(ann.run([ 1, 1 ]))).to.deep.equal([ 1, 1, 0, 0, 0 ]);
				expect(booleanThreshold(ann.run([ 1, 0 ]))).to.deep.equal([ 0, 1, 1, 0, 1 ]);
			});
	});

	describe('Options Tests', function() {
		var optionsToTest = annOptionsSchema.getData().properties;

		var stringTest = function(optionToTest, validValues) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });

			var initalValue = ann.getOption(optionToTest);
			expect(initalValue).to.be.a('string');

			// Test all valid values
			for (var testValue of validValues) {
				ann.setOption(optionToTest, testValue);
				var updatedValue = ann.getOption(optionToTest);
				expect(updatedValue).to.equal(testValue);
			}
		};

		var numberTest = function(optionToTest, min, max) {
			// List of FANN methods known to fail to update value in a way that is observable
			var numberExceptions = {
				sarpropWeightDecayShift: true,
				sarpropStepErrorThresholdFactor: true,
				sarpropStepErrorShift: true,
				sarpropTemperature: true
			};

			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var testIncrement = 1;
			var hasMax = (typeof max === 'number');
			var hasMin = (typeof min === 'number');

			// Determine value to set setter with based on min, max and the intial value.
			if (hasMax && hasMin && ((max - min) <= 1) ) testIncrement = 0.1;
			if (max <= testIncrement) testIncrement = 0.1;
			if (initalValue === max) testIncrement = 0 - testIncrement;
			var testValue = initalValue + testIncrement;

			var setTestFunc = function() { ann.setOption(optionToTest, testValue); }
			expect(initalValue).to.be.a('number');
			expect(setTestFunc).to.not.throw();

			// If a optionToTest is a known exception don't test the value.
			if (numberExceptions[optionToTest]) return;

			var updatedValue = ann.getOption(optionToTest);
			expect(updatedValue).to.be.closeTo(testValue, 1e-7);
		};

		var arrayTest = function(optionToTest, validArray) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var setTestFunc = function() { ann.setOption(optionToTest, validArray); }
			expect(initalValue).to.be.an('array');
			expect(setTestFunc).to.not.throw();
			var updatedArray = ann.getOption(optionToTest);
			if (typeof validArray[0] === 'number') expect(updatedArray).to.eql(validArray);
			if (typeof validArray[0] === 'string') {
				for (var i = 0; i < validArray.length; i++) {
					expect(updatedArray[i]).to.include(validArray[i]);
				}
			}
		};

		var optionTestRunner = function(optionToTest) {
			it(optionToTest + ' has working getters and setters', function() {
				expect(optionsToTest).to.have.property(optionToTest);
				var testSettings = optionsToTest[optionToTest];
				expect(testSettings).to.have.property('type');
				var testType = testSettings.type;
				switch (testType) {
					case 'string':
						stringTest(optionToTest, testSettings.enum);
						break;
					case 'number':
						numberTest(optionToTest, testSettings.min, testSettings.max);
						break;
					case 'array':
						var testArray =(testSettings.elements.type === 'string') ? testSettings.elements.enum : [ 0, 1, 2 ];
						arrayTest(optionToTest, testArray);
						break;
					default:
						expect(true).to.be.false;
				}
			});
		}

		for (var optionToTest in optionsToTest) {
			optionTestRunner(optionToTest);
		}
	});

	describe('Randomize Weights', function() {
		it('Can call randomizeWeights', function() {
			var ann = createANN({ layers: [ 2, 1, 2 ] });
			var initalConnections = ann.getConnectionArray();
			ann.randomizeWeights(-5, 5);
			var updatedConnections = ann.getConnectionArray();
			expect(initalConnections).to.be.an('array');
			expect(updatedConnections).to.be.an('array');
			expect(updatedConnections).to.have.lengthOf(initalConnections.length);
			for (var i = 0; i < updatedConnections.length; i++) {
				expect(updatedConnections[i]).to.have.property('from_neuron').that.equals(initalConnections[i].from_neuron);
				expect(updatedConnections[i]).to.have.property('to_neuron').that.equals(initalConnections[i].to_neuron);
				expect(updatedConnections[i]).to.have.property('weight').that.is.not.equal(initalConnections[i].weight);
			}
		});
	});

	describe('init Weights', function() {
		it('Can call initWeights', function() {
			var ann = createANN({ layers: [ 2, 1, 5 ] });
			var initalConnections = ann.getConnectionArray();
			var td = createTrainingData(booleanTrainingData);
			ann.initWeights(td);
			var updatedConnections = ann.getConnectionArray();
			expect(initalConnections).to.be.an('array');
			expect(updatedConnections).to.be.an('array');
			expect(updatedConnections).to.have.lengthOf(initalConnections.length);
			for (var i = 0; i < updatedConnections.length; i++) {
				expect(updatedConnections[i]).to.have.property('from_neuron').that.equals(initalConnections[i].from_neuron);
				expect(updatedConnections[i]).to.have.property('to_neuron').that.equals(initalConnections[i].to_neuron);
				expect(updatedConnections[i]).to.have.property('weight').that.is.not.equal(initalConnections[i].weight);
			}
		});
	});

	describe('Print Connections', function() {
		it('Can call printConnections', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var testFunc = function() { ann.printConnections(); }
			expect(testFunc).to.not.throw();
		});
	});

	describe('Get MSE', function() {
		it('Can call getMSE', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var mse = ann.getMSE();
			expect(mse).to.be.a('number');
		});
	});

	describe('Reset MSE', function() {
		it('Can call resetMSE', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var testFunc = function() { ann.resetMSE(); }
			expect(testFunc).to.not.throw();
		});
	});
});
