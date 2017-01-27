var expect = require('chai').expect;
var fanny = require('../lib');

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
		expect(td).to.exist;
		expect(td._fannyTrainingData).to.exist;
		expect(td._datatype).to.exist;
		expect(td._datatype).to.equal('float');
	});
	describe('prototype functions', function() {
		beforeEach(function() {
			this.td = createTrainingData(booleanTrainingData, 'float');
		});
		it('should return the length of the training data', function() {
			var length = this.td.getLength();
			expect(length).to.equal(booleanTrainingData.length);
		});
		it('should return the number of inputs', function() {
			expect(this.td.getNumInputs()).to.equal(2);
		});
		it('should return the number of outputs', function() {
			expect(this.td.getNumOutputs()).to.equal(5);
		});
		it('should return the inputs', function() {
			expect(this.td.getInputData()).to.deep.equal(booleanInputData);
		});
		it('should return the outputs', function() {
			expect(this.td.getOutputData()).to.deep.equal(booleanOutputData);
		});
		it('should return an input at the specified position', function() {
			expect(this.td.getOneInputData(3)).to.deep.equal(booleanInputData[3]);
		});
		it('should return an output at the specified position', function() {
			expect(this.td.getOneOutputData(3)).to.deep.equal(booleanOutputData[3]);
		});
		it('should return the min input', function() {
			expect(this.td.getMinInput()).to.equal(0);
		});
		it('should return the max input', function() {
			expect(this.td.getMaxInput()).to.equal(1);
		});
		it('should return the min output', function() {
			expect(this.td.getMinOutput()).to.equal(0);
		});
		it('should return the max output', function() {
			expect(this.td.getMaxOutput()).to.equal(1);
		});
		it('should reset the max and min input', function() {
			var min = -1;
			var max = 2;
			this.td.scaleInput(min, max);
			expect(this.td.getMinInput()).to.equal(min);
			expect(this.td.getMaxInput()).to.equal(max);
		});
		it('should reset the max and min output', function() {
			var min = -1;
			var max = 2;
			this.td.scaleOutput(min, max);
			expect(this.td.getMinOutput()).to.equal(min);
			expect(this.td.getMaxOutput()).to.equal(max);
		});
		it('should scale the input and output', function() {
			var min = -1;
			var max = 2;
			this.td.scale(min, max);
			expect(this.td.getMinInput()).to.equal(min);
			expect(this.td.getMaxInput()).to.equal(max);
			expect(this.td.getMinOutput()).to.equal(min);
			expect(this.td.getMaxOutput()).to.equal(max);
		});
		it('should return a subset of the training data', function() {
			this.td.subset(1, 3);
			expect(this.td.getInputData()).to.deep.equal(booleanInputData.slice(1, 4));
			expect(this.td.getOutputData()).to.deep.equal(booleanOutputData.slice(1, 4));
		});
		it('should merge two training datas together', function() {
			var data = [
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ],
				[ [ 1, 0 ], [ 0, 1, 1, 0, 1 ] ]
			];
			var td2 = createTrainingData(data);
			this.td.merge(td2);
			expect(this.td.getLength()).to.equal(data.length + booleanTrainingData.length);
		});
		it('should shuffle the inputs', function() {
			var inputData =  this.td.getOneInputData(0);
			var outputData = this.td.getOutputData(0);
			expect(inputData).to.exist;
			expect(outputData).to.exist;
			this.td.shuffle();
			expect(this.td.getOneInputData(0)).to.not.deep.equal(inputData);
			expect(this.td.getOneOutputData(0)).to.not.deep.equal(outputData);
		});
	});
});
