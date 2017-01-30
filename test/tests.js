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

<<<<<<< Updated upstream
=======
	describe('Options Tests', function() {
		var optionsToTest = annOptionsSchema.toJSONSchema().properties;

		var stringTest = function(optionToTest, validValues) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var testValue = (initalValue === validValues[0]) ? validValues[1] : validValues[0];
			ann.setOption(optionToTest, testValue);
			var setTestFunc = function() { ann.setOption(optionToTest, testValue); }
			expect(initalValue).to.be.a('string');
			expect(setTestFunc).to.not.throw();
		};

		var numberTest = function(optionToTest) {
			var ann = createANN({ layers: [ 2, 20, 5 ] });
			var initalValue = ann.getOption(optionToTest);
			var testValue = initalValue + 0.1;
			var setTestFunc = function() { ann.setOption(optionToTest, testValue); }
			expect(initalValue).to.be.a('number');
			expect(setTestFunc).to.not.throw();
		};

		var arrayTest = function(optionToTest, validArray) {
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
					case 'string':
						stringTest(optionToTest, testSettings.enum);
						break;
					case 'number':
						numberTest(optionToTest);
						break;
					case 'array':
						var testArray;
						if (testSettings.items.type === String) testArray = testSettings.items.enum
						else if (testSettings.items.type === Number) testArray = [ 0, 1, 2 ];
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

	describe.only('Print Connections', function() {
		var ann = createANN({ layers: [ 2, 20, 5 ] });
		ann.printConnections();
	})
>>>>>>> Stashed changes
});
