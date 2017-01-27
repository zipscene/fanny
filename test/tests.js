var expect = require('chai').expect;
var fanny = require('../lib');

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
		var optionsToTest = {
			trainingAlgorithm: {
				type: String,
				enum: [ 'INCREMENTAL', 'BATCH', 'RPROP', 'QUICKPROP', 'SARPROP' ]
			},
			learningRate: {
				type: Number
			},
			trainErrorFunction: {
				type: String,
				enum: [ 'LINEAR', 'TANH' ]
			},
			quickpropDecay: {
				type: Number
			},
			quickpropMu: {
				type: Number
			},
			rpropIncreaseFactor: {
				type: Number
			},
			rpropDecreaseFactor: {
				type: Number
			},
			rpropDeltaZero: {
				type: Number
			},
			rpropDeltaMin: {
				type: Number
			},
			rpropDeltaMax: {
				type: Number
			},
			sarpropWeightDecayShift: {
				type: Number
			},
			sarpropStepErrorThresholdFactor: {
				type: Number
			},
			sarpropStepErrorShift: {
				type: Number
			},
			sarpropTemperature: {
				type: Number
			},
			learningMomentum: {
				type: Number
			},
			trainStopFunction: {
				type: String,
				enum: [ 'BIT', 'MSE' ]
			},
			bitFailLimit: {
				type: Number
			},
			cascadeOutputChangeFraction: {
				type: Number
			},
			cascadeOutputStagnationEpochs: {
				type: Number
			},
			cascadeCandidateChangeFraction: {
				type: Number
			},
			cascadeCandidateStagnationEpochs: {
				type: Number
			},
			cascadeWeightMultiplier: {
				type: Number
			},
			cascadeCandidateLimit: {
				type: Number
			},
			cascadeMaxOutEpochs: {
				type: Number
			},
			cascadeMaxCandEpochs: {
				type: Number
			},
			cascadeActivationFunctions: {
				type: 'array',
				elements: {
					type: String,
					required: true,
					enum: [
						'LINEAR', 'THRESHOLD', 'THRESHOLD_SYMMETRIC', 'SIGMOID', 'SIGMOID_STEPWISE',
						'SIGMOID_SYMMETRIC', 'SIGMOID_SYMMETRIC_STEPWISE', 'GAUSSIAN', 'GAUSSIAN_SYMMETRIC',
						'ELLIOT', 'ELLIOT_SYMMETRIC', 'LINEAR_PIECE', 'LINEAR_PIECE_SYMMETRIC', 'SIN_SYMMETRIC',
						'COS_SYMMETRIC'
					]
				}
			},
			cascadeActivationSteepnesses: {
				type: 'array',
				elements: {
					type: Number
				}
			},
			cascadeNumCandidateGroups: {
				type: Number
			}
		};

		var stringTest = function(optionToTest, validValues, invalidValue) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var testValue = (initalValue === validValues[0]) ? validValues[1] : validValues[0];
			ann.setOption(optionToTest, testValue);
			var setTestFunc = function() { ann.setOption(optionToTest, testValue); }
			expect(initalValue).to.be.a('string');
			expect(setTestFunc).to.not.throw();
		};

		var numberTest = function(optionToTest, lowerValidNumber, upperValidNumber) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var testValue = initalValue + 0.1;
			var setTestFunc = function() { ann.setOption(optionToTest, testValue); }
			expect(initalValue).to.be.a('number');
			expect(setTestFunc).to.not.throw();
		};

		var arrayTest = function(optionToTest, validArray, inValidArray) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var setTestFunc = function() { ann.setOption(optionToTest, validArray); }
			expect(initalValue).to.be.an('array');
			expect(setTestFunc).to.not.throw();
		};

		var optionTestRunner = function(optionToTest) {
			it(optionToTest + ' has working getters and setters', function() {
				expect(optionsToTest).to.have.property(optionToTest);
				var testSettings = optionsToTest[optionToTest];
				expect(testSettings).to.have.property('type');
				var testType = testSettings.type;
				switch (testType) {
					case String:
						stringTest(optionToTest, testSettings.enum, 'I am not a valid fann enum anywhere');
						break;
					case Number:
						numberTest(optionToTest);
						break;
					case 'array':
						var testArray;
						if (testSettings.elements.type === String) testArray = testSettings.elements.enum
						else if (testSettings.elements.type === Number) testArray = [ 0, 1, 2 ];
						arrayTest(optionToTest, testArray, ['I am not a valid value anywhere']);
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

});
