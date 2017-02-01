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

	describe('Print Commands', function() {
		it('Can call printConnections', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var testFunc = function() { ann.printConnections(); }
			expect(testFunc).to.not.throw();
		});
		it.skip('Can call printParameters', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var testFunc = function() { ann.printParameters(); }
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
	describe('Get Activation Function', function() {
		it('Can call getActivationFunction', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var activationFunction = ann.getActivationFunction(1, 1);
			expect(activationFunction).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
		});
		it('Returns null for calls outside the network', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var activationFunction = ann.getActivationFunction(3, 3);
			expect(activationFunction).to.be.null;
		});
	});

	describe('Set Activation Function', function() {
		it('Can call setActivationFunction', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var initialActivationFunction = ann.getActivationFunction(1, 1);
			expect(initialActivationFunction).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			ann.setActivationFunction('LINEAR', 1, 1);
			var updatedActivationFunction = ann.getActivationFunction(1, 1);
			expect(updatedActivationFunction).to.be.a('string').and.to.equal('LINEAR');
		});

		it('Can set to all ActivationFunction enum values', function() {
			var activationFunctions = [
				'LINEAR', 'THRESHOLD', 'THRESHOLD_SYMMETRIC', 'SIGMOID', 'SIGMOID_STEPWISE',
				'SIGMOID_SYMMETRIC', 'SIGMOID_SYMMETRIC_STEPWISE', 'GAUSSIAN', 'GAUSSIAN_SYMMETRIC',
				'ELLIOT', 'ELLIOT_SYMMETRIC', 'LINEAR_PIECE', 'LINEAR_PIECE_SYMMETRIC', 'SIN_SYMMETRIC',
				'COS_SYMMETRIC'
			];

			var ann = createANN({ layers: [ 2, 2, 2 ] });
			// Test all enum values
			for (var activationFunction of activationFunctions) {
				ann.setActivationFunction(activationFunction, 1, 1);
				var setActivationFunction = ann.getActivationFunction(1, 1);
				expect(setActivationFunction).to.equal(activationFunction);
			}
		});

		it('Calls outside the network have no effect', function() {
			var ann = createANN({ layers: [ 1, 1, 1 ] });
			var initalActivationFunction1 = ann.getActivationFunction(1, 1);
			var initalActivationFunction2 = ann.getActivationFunction(2, 1);
			ann.setActivationFunction('LINEAR_PIECE', 3, 1);
			expect(ann.getActivationFunction(1, 1)).to.be.a('string').and.to.equal(initalActivationFunction1);
			expect(ann.getActivationFunction(2, 1)).to.be.a('string').and.to.equal(initalActivationFunction2);
		});

		it('Can call setActivationFunctionLayer', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			// Setup state of functions
			ann.setActivationFunction('SIGMOID_STEPWISE', 1, 1);
			ann.setActivationFunction('SIGMOID_STEPWISE', 1, 2);
			var initialActivationFunction1 = ann.getActivationFunction(1, 1);
			var initialActivationFunction2 = ann.getActivationFunction(1, 2);
			expect(initialActivationFunction1).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			expect(initialActivationFunction2).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			ann.setActivationFunctionLayer('LINEAR', 1);
			var updatedActivationFunction1 = ann.getActivationFunction(1, 1);
			var updatedActivationFunction2 = ann.getActivationFunction(1, 2);
			expect(updatedActivationFunction1).to.be.a('string').and.to.equal('LINEAR');
			expect(updatedActivationFunction2).to.be.a('string').and.to.equal('LINEAR');
		});

		it('Can call setActivationFunctionHidden', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			// Setup state of functions
			ann.setActivationFunction('SIGMOID_STEPWISE', 1, 1);
			ann.setActivationFunction('SIGMOID_STEPWISE', 1, 2);
			var initialActivationFunction1 = ann.getActivationFunction(1, 1);
			var initialActivationFunction2 = ann.getActivationFunction(1, 2);
			expect(initialActivationFunction1).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			expect(initialActivationFunction2).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			ann.setActivationFunctionHidden('LINEAR');
			var updatedActivationFunction1 = ann.getActivationFunction(1, 1);
			var updatedActivationFunction2 = ann.getActivationFunction(1, 2);
			expect(updatedActivationFunction1).to.be.a('string').and.to.equal('LINEAR');
			expect(updatedActivationFunction2).to.be.a('string').and.to.equal('LINEAR');
		});

		it('Can call setActivationFunctionOutput', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			// Setup state of functions
			ann.setActivationFunction('SIGMOID_STEPWISE', 2, 1);
			ann.setActivationFunction('SIGMOID_STEPWISE', 2, 2);
			var initialActivationFunction1 = ann.getActivationFunction(2, 1);
			var initialActivationFunction2 = ann.getActivationFunction(2, 2);
			expect(initialActivationFunction1).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			expect(initialActivationFunction2).to.be.a('string').and.to.equal('SIGMOID_STEPWISE');
			ann.setActivationFunctionOutput('LINEAR');
			var updatedActivationFunction1 = ann.getActivationFunction(2, 1);
			var updatedActivationFunction2 = ann.getActivationFunction(2, 2);
			expect(updatedActivationFunction1).to.be.a('string').and.to.equal('LINEAR');
			expect(updatedActivationFunction2).to.be.a('string').and.to.equal('LINEAR');
		});
	});
	describe('#getBiasArray', function() {
		it('should return an array', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			expect(ann.getBiasArray()).to.be.instanceof(Array);
		});
	});
	describe('#getLayerArray', function() {
		it('should return an array', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			expect(ann.getLayerArray()).to.be.instanceof(Array);
		});
	});
	describe('#scaleTrainingData', function() {
		it('can throw an error', function() {
			var ann = createANN({ layers: [ 2, 2, 2 ] });
			var func = function() {
				return ann.scaleTrainingData([]);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can call scale train data', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.scaleTrainingData(data);
			}).to.not.throw();
		});
	});
	describe('#descaleTrainingData', function() {
		it('can throw an error', function() {
			var func = function() {
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.descaleTrainingData([]);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can call descale train data', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.descaleTrainingData(data);
			}).to.not.throw();
		});
	});
	describe('#setInputScalingParams', function() {
		it('can throw an error if data is not trainingData', function() {
			var func = function() {
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setInputScalingParams([], 0, 1);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can throw an error if min is not a number', function() {
			var func = function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setInputScalingParams(data, 'blah', 1);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can throw an error if max is not a number', function() {
			var func = function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setInputScalingParams(data, 0, 'blah');
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can call setInputScalingParams', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				return ann.setInputScalingParams(data, 0, 1);
			}).to.not.throw();
		});
	});
	describe('#setOutputScalingParams', function() {
		it('can throw an error if data is not trainingData', function() {
			var func = function() {
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setOutputScalingParams([], 0, 1);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can throw an error if min is not a number', function() {
			var func = function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setOutputScalingParams(data, 'blah', 1);
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can throw an error if max is not a number', function() {
			var func = function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 2, 2 ] });
				return ann.setOutputScalingParams(data, 0, 'blah');
			};
			expect(func).to.throw(XError.INVALID_ARGUMENT);
		});
		it('can call setOutputScalingParams', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				return ann.setOutputScalingParams(data, 0, 1);
			}).to.not.throw();
		});
	});
	describe('#setScalingParams', function() {
		it('can call setScalingParams', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				return ann.setScalingParams(data, 0, 1, 0, 1);
			}).to.not.throw();
		});
	});
	describe('#clearScalingParams', function() {
		it('can clear scaling params', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.clearScalingParams();
			}).to.not.throw();
		});
	});
	describe('#scaleInput', function() {
		it('can scale input', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.scaleInput([ 0, 1 ]);
			}).to.not.throw();
		});
	});
	describe('#scaleOutput', function() {
		it('can scale output', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.scaleOutput([ 0, 1, 1, 1, 0 ]);
			}).to.not.throw();
		});
	});
	describe('#descaleInput', function() {
		it('can descale input', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.descaleInput([ 0, 1 ]);
			}).to.not.throw();
		});
	});
	describe('#descaleOutput', function() {
		it('can descale output', function() {
			expect(function() {
				var data = createTrainingData(booleanTrainingData);
				var ann = createANN({ layers: [ 2, 1, 5 ] });
				ann.setScalingParams(data, 0, 1, 0, 1);
				return ann.descaleOutput([ 0, 1, 1, 1, 0 ]);
			}).to.not.throw();
		});
	});
	describe('#getActivationSteepness', function() {
		it('can get activation steepness', function() {
			var data = createTrainingData(booleanTrainingData);
			var ann = createANN({ layers: [ 2, 1, 5 ] });
			var steepness = ann.getActivationSteepness(1, 1);
			expect(steepness).to.be.a('number');
			expect(steepness).to.not.equal(-1);
		});
	});
});
