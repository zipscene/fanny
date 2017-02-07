var expect = require('chai').expect;
var fanny = require('../lib');
var zstreams = require('zstreams');
var fs = require('fs');
var XError = require('xerror');

var createTrainingData = fanny.createTrainingData;
var loadTrainingData = fanny.loadTrainingData;

// Inputs: A, B  Outputs: AND, OR, NAND, NOR, XOR
var booleanTrainingData = [
	[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ],
	[ [ 0, 1 ], [ 0, 1, 1, 0, 1 ] ],
	[ [ 0, 0 ], [ 0, 0, 1, 1, 0 ] ],
	[ [ 1, 1 ], [ 1, 1, 0, 0, 0 ] ]
];

var booleanInputData = [
	[ 1, 0 ],
	[ 0, 1 ],
	[ 0, 0 ],
	[ 1, 1 ],
];

var booleanOutputData = [
	[ 0, 1, 1, 0, 1 ],
	[ 0, 1, 1, 0, 1 ],
	[ 0, 0, 1, 1, 0 ],
	[ 1, 1, 0, 0, 0 ]
];

describe('Training Data', function() {
	it('#createTrainingData', function() {
		var td = createTrainingData(booleanTrainingData, 'float');
		var td2 = createTrainingData(booleanInputData, booleanOutputData, 'float');
		expect(td._fannyTrainingData).to.exist;
		expect(td._datatype).to.exist;
		expect(td._datatype).to.equal('float');
		expect(td2._fannyTrainingData).to.exist;
		expect(td2._datatype).to.exist;
		expect(td2._datatype).to.equal('float');
	});
	it('#loadTrainingData', function() {
		return loadTrainingData('test/resources/training-data.txt', 'float')
			.then((td) => {
				expect(td._fannyTrainingData).to.exist;
				expect(td._datatype).to.exist;
				expect(td._datatype).to.equal('float');
			});
	});
	describe('prototype functions', function() {
		it('#getLength', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			var length = td.getLength();
			expect(length).to.equal(booleanTrainingData.length);
		});
		it('#getNumInputs', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getNumInputs()).to.equal(2);
		});
		it('#getNumOutputs', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getNumOutputs()).to.equal(5);
		});
		it('#getInputData', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getInputData()).to.deep.equal(booleanInputData);
		});
		it('#getOutputData', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getOutputData()).to.deep.equal(booleanOutputData);
		});
		it('#getOneInputData', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getOneInputData(3)).to.deep.equal(booleanInputData[3]);
		});
		it('#getOneInputData error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.getOneInputData('3');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#getOneOutputData', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getOneOutputData(3)).to.deep.equal(booleanOutputData[3]);
		});
		it('#getOneOutputData error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.getOneOutputData('3');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#getMinInput', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getMinInput()).to.equal(0);
		});
		it('#getMaxInput', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getMaxInput()).to.equal(1);
		});
		it('#getMinOutput', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getMinOutput()).to.equal(0);
		});
		it('#getMaxOutput', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			expect(td.getMaxOutput()).to.equal(1);
		});
		it('#scaleInput', function() {
			var min = -1;
			var max = 2;
			var td = createTrainingData(booleanTrainingData, 'float');
			td.scaleInput(min, max);
			expect(td.getMinInput()).to.equal(min);
			expect(td.getMaxInput()).to.equal(max);
		});
		it('#scaleInput Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scaleInput('3', 1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scaleInput(3, '1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#scaleOutput', function() {
			var min = -1;
			var max = 2;
			var td = createTrainingData(booleanTrainingData, 'float');
			td.scaleOutput(min, max);
			expect(td.getMinOutput()).to.equal(min);
			expect(td.getMaxOutput()).to.equal(max);
		});
		it('#scaleOutput Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scaleOutput('1', -1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scaleOutput(1, '-1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#scale', function() {
			var min = -1;
			var max = 2;
			var td = createTrainingData(booleanTrainingData, 'float');
			td.scale(min, max);
			expect(td.getMinInput()).to.equal(min);
			expect(td.getMaxInput()).to.equal(max);
			expect(td.getMinOutput()).to.equal(min);
			expect(td.getMaxOutput()).to.equal(max);
		});
		it('#scale Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scale('1', -1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.scale(1, '-1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#subset', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			td.subset(1, 3);
			expect(td.getInputData()).to.deep.equal(booleanInputData.slice(1, 4));
			expect(td.getOutputData()).to.deep.equal(booleanOutputData.slice(1, 4));
		});
		it('#subset Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.subset('1', 1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.subset(1, '1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#merge', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			var data = [
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ],
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ]
			];
			var td2 = createTrainingData(data);
			td.merge(td2);
			expect(td.getLength()).to.equal(data.length + booleanTrainingData.length);
		});
		it('#merge Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.merge([ [ 1, 0 ], [ 0 , 1, 1, 0, 1]]);
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#shuffle', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			var inputData =  td.getOneInputData(0);
			var outputData = td.getOutputData(0);
			expect(inputData).to.exist;
			expect(outputData).to.exist;
			td.shuffle();
			expect(td.getOneInputData(0)).to.not.deep.equal(inputData);
			expect(td.getOneOutputData(0)).to.not.deep.equal(outputData);
		});
		it('#clone', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			var tdClone = td.clone();
			expect(tdClone.getInputData()).to.deep.equal(td.getInputData());
			expect(tdClone.getOutputData()).to.deep.equal(td.getOutputData());
		});
		it('#setData', function() {
			var inputData = [ [ 0, 1 ] ];
			var outputData = [ [ 1, 0, 1, 0, 0 ] ];
			var td = createTrainingData(booleanTrainingData, 'float');
			td.setData(inputData, outputData);
			expect(td.getInputData()).to.deep.equal(inputData);
			expect(td.getOutputData()).to.deep.equal(outputData);
		});
		it('#setData by object', function() {
			var input = [ [ 0, 1 ] ];
			var output = [ [ 1, 0, 0, 0, 1 ] ];
			var data = [
				{
					input: input[0],
					output: output[0]
				}
			];
			var td = createTrainingData(booleanTrainingData, 'float');
			td.setData(data);
			expect(td.getInputData()).to.deep.equal(input);
			expect(td.getOutputData()).to.deep.equal(output);
		});
		it('#setData Error', function() {
			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.setData([ 0, 1 ], [ 1, 0, 1, 0, 0 ]);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				var td = createTrainingData(booleanTrainingData, 'float');
				return td.subset({ input: [ 0, 1 ] });
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#save', function() {
			var td = createTrainingData(booleanTrainingData, 'float');
			var filename = 'test/test-bool-data.txt';
			return td.save(filename)
				.then(function() {
					return zstreams.fromFile(filename)
						.split('\n')
						.through((data) => {
							return data
								.trim()
								.split(/\s/g)
								.map(function(d) { return parseInt(d, 10) });
						})
						.intoArray()
				})
				.then(function(array) {
					var input = [ array[1], array[3], array[5], array[7] ];
					var output = [ array[2], array[4], array[6], array[8] ];
					expect(input).to.deep.equal(booleanInputData);
					expect(output).to.deep.equal(booleanOutputData);
					return fs.unlinkSync(filename);
				}, (err) => {
					fs.unlinkSync(filename);
					throw err;
				});
		});
	});
});
