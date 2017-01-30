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

describe.only('Training Data', function() {
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
		return loadTrainingData('test/examples/training-data.txt', 'float')
			.then((td) => {
				expect(td._fannyTrainingData).to.exist;
				expect(td._datatype).to.exist;
				expect(td._datatype).to.equal('float');
			});
	});
	describe('prototype functions', function() {
		beforeEach(function() {
			this.td = createTrainingData(booleanTrainingData, 'float');
		});
		after(function() {
			if (this.filename) {
				fs.unlinkSync(this.filename);
			}
		});
		it('#getLength', function() {
			var length = this.td.getLength();
			expect(length).to.equal(booleanTrainingData.length);
		});
		it('#getNumInputs', function() {
			expect(this.td.getNumInputs()).to.equal(2);
		});
		it('#getNumOutputs', function() {
			expect(this.td.getNumOutputs()).to.equal(5);
		});
		it('#getInputData', function() {
			expect(this.td.getInputData()).to.deep.equal(booleanInputData);
		});
		it('#getOutputData', function() {
			expect(this.td.getOutputData()).to.deep.equal(booleanOutputData);
		});
		it('#getOneInputData', function() {
			expect(this.td.getOneInputData(3)).to.deep.equal(booleanInputData[3]);
		});
		it('#getOneInputData error', function() {
			var self = this;
			expect(function() {
				return self.td.getOneInputData('3');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#getOneOutputData', function() {
			expect(this.td.getOneOutputData(3)).to.deep.equal(booleanOutputData[3]);
		});
		it('#getOneOutputData error', function() {
			var self = this;
			expect(function() {
				return self.td.getOneOutputData('3');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#getMinInput', function() {
			expect(this.td.getMinInput()).to.equal(0);
		});
		it('#getMaxInput', function() {
			expect(this.td.getMaxInput()).to.equal(1);
		});
		it('#getMinOutput', function() {
			expect(this.td.getMinOutput()).to.equal(0);
		});
		it('#getMaxOutput', function() {
			expect(this.td.getMaxOutput()).to.equal(1);
		});
		it('#scaleInput', function() {
			var min = -1;
			var max = 2;
			this.td.scaleInput(min, max);
			expect(this.td.getMinInput()).to.equal(min);
			expect(this.td.getMaxInput()).to.equal(max);
		});
		it('#scaleInput Error', function() {
			var self = this;
			expect(function() {
				return self.td.scaleInput('3', 1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				return self.td.scaleInput(3, '1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#scaleOutput', function() {
			var min = -1;
			var max = 2;
			this.td.scaleOutput(min, max);
			expect(this.td.getMinOutput()).to.equal(min);
			expect(this.td.getMaxOutput()).to.equal(max);
		});
		it('#scaleOutput Error', function() {
			var self = this;
			expect(function() {
				return self.td.scaleOutput('1', -1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				return self.td.scaleOutput(1, '-1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#scale', function() {
			var min = -1;
			var max = 2;
			this.td.scale(min, max);
			expect(this.td.getMinInput()).to.equal(min);
			expect(this.td.getMaxInput()).to.equal(max);
			expect(this.td.getMinOutput()).to.equal(min);
			expect(this.td.getMaxOutput()).to.equal(max);
		});
		it('#scale Error', function() {
			var self = this;
			expect(function() {
				return self.td.scale('1', -1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				return self.td.scale(1, '-1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#subset', function() {
			this.td.subset(1, 3);
			expect(this.td.getInputData()).to.deep.equal(booleanInputData.slice(1, 4));
			expect(this.td.getOutputData()).to.deep.equal(booleanOutputData.slice(1, 4));
		});
		it('#subset Error', function() {
			var self = this;
			expect(function() {
				return self.td.subset('1', 1);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				return self.td.subset(1, '1');
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#merge', function() {
			var data = [
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ],
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ]
			];
			var td2 = createTrainingData(data);
			this.td.merge(td2);
			expect(this.td.getLength()).to.equal(data.length + booleanTrainingData.length);
		});
		it('#merge Error', function() {
			var self = this;
			expect(function() {
				return self.td.merge([ [ 1, 0 ], [ 0 , 1, 1, 0, 1]]);
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#shuffle', function() {
			var inputData =  this.td.getOneInputData(0);
			var outputData = this.td.getOutputData(0);
			expect(inputData).to.exist;
			expect(outputData).to.exist;
			this.td.shuffle();
			expect(this.td.getOneInputData(0)).to.not.deep.equal(inputData);
			expect(this.td.getOneOutputData(0)).to.not.deep.equal(outputData);
		});
		it('#clone', function() {
			var tdClone = this.td.clone();
			expect(tdClone.getInputData()).to.deep.equal(this.td.getInputData());
			expect(tdClone.getOutputData()).to.deep.equal(this.td.getOutputData());
		});
		it('#setData', function() {
			var inputData = [ [ 0, 1 ] ];
			var outputData = [ [ 1, 0, 1, 0, 0 ] ];
			this.td.setData(inputData, outputData);
			expect(this.td.getInputData()).to.deep.equal(inputData);
			expect(this.td.getOutputData()).to.deep.equal(outputData);
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
			this.td.setData(data);
			expect(this.td.getInputData()).to.deep.equal(input);
			expect(this.td.getOutputData()).to.deep.equal(output);
		});
		it('#setData Error', function() {
			var self = this;
			expect(function() {
				return self.td.setData([ 0, 1 ], [ 1, 0, 1, 0, 0 ]);
			}).to.throw(XError.INVALID_ARGUMENT);

			expect(function() {
				return self.td.subset({ input: [ 0, 1 ] });
			}).to.throw(XError.INVALID_ARGUMENT);
		});
		it('#save', function() {
			var self = this;
			self.filename = 'test/test-bool-data.txt';
			return self.td.save(self.filename)
				.then(function() {
					return zstreams.fromFile(self.filename)
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
				});
		});
	});
});
