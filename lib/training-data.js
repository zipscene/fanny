var utils = require('./utils');
var XError = require('xerror');

function TrainingData(fannyTrainingData, datatype) {
	this._fannyTrainingData = fannyTrainingData;
	this._datatype = datatype;
}

TrainingData.prototype.clone = function() {
	var addon = utils.getAddon(this._datatype);
	var fannyTrainingData = new addon.TrainingData(this._fannyTrainingData);
	return new TrainingData(fannyTrainingData, this._datatype);
};

// Possible invocations:
// setData(<ArrayOfPairs>)
// setData(<InputsArrayOfArrays>, <OutputsArrayOfArrays>)
// For ArrayOfPairs, it should be an array of objects.  Each object should
// contain the keys "input" and "output", and each should be an array of
// the appropriate size.
TrainingData.prototype.setData = function(arg1, arg2) {
	var inputs = [];
	var outputs = [];
	var i, entry;
	if (Array.isArray(arg1) && !arg2) {
		for (i = 0; i < arg1.length; ++i) {
			entry = arg1[i];
			if (Array.isArray(entry) && entry.length === 2 && Array.isArray(entry[0]) && Array.isArray(entry[1])) {
				inputs.push(entry[0]);
				outputs.push(entry[1]);
			} else if (typeof entry === 'object' && entry && Array.isArray(entry.input) && Array.isArray(entry.output)) {
				inputs.push(entry.input);
				inputs.push(entry.output);
			} else {
				throw new XError(XError.INVALID_ARGUMENT);
			}
		}
	} else if (Array.isArray(arg1) && Array.isArray(arg2) && Array.isArray(arg1[0])) {
		if (arg1.length !== arg2.length) throw new XError(XError.INVALID_ARGUMENT);
		for (i = 0; i < arg1.length; ++i) {
			if (!Array.isArray(arg1[i]) || !Array.isArray(arg2[i])) throw new XError(XError.INVALID_ARGUMENT);
			inputs.push(arg1[i]);
			outputs.push(arg2[i]);
		}
	} else {
		throw new XError(XError.INVALID_ARGUMENT);
	}
	try {
		this._fannyTrainingData.setTrainData(inputs, outputs);
	} catch (ex) {
		throw new XError(XError.INVALID_ARGUMENT, ex);
	}
};

// If fixedDecimalPoint is set, it's saved to a fixed format
TrainingData.prototype.save = function(filename, fixedDecimalPoint) {
	var self = this;
	if (typeof filename !== 'string') throw new XError(XError.INVALID_ARGUMENT);
	return new Promise(function(resolve, reject) {
		var cb = function(err) {
			if (err) return reject(new XError(err));
			resolve();
		};
		if (typeof fixedDecimalPoint === 'number') {
			self._fannyTrainingData.saveTrainToFixed(filename, fixedDecimalPoint, cb);
		} else {
			self._fannyTrainingData.saveTrain(filename, cb);
		}
	});
};

/* Need additional prototype methods:
 * - shuffle()
 * - merge()
 * - getLength()
 * - getNumInputs()
 * - getNumOutputs()
 * - getInputData()
 * - getOutputData()
 * - getOneInputData()
 * - getOneOutputData()
 * - getMinInput()
 * - getMaxInput()
 * - getMinOutput()
 * - getMaxOutput()
 * - scaleInput()
 * - scaleOutput()
 * - scale()
 * - subset()
*/

function createTrainingData(arg1, arg2, datatype) {
	if (typeof arg2 === 'string') {
		datatype = arg2;
		arg2 = undefined;
	}
	if (!datatype) datatype = 'float';
	var addon = utils.getAddon(datatype);
	var fannyTrainingData = new addon.TrainingData();
	var td = new TrainingData(fannyTrainingData, datatype);
	td.setData(arg1, arg2);
	return td;
}

function loadTrainingData(filename, datatype) {
	if (!filename) throw new XError(XError.INVALID_ARGUMENT, 'filename is required');
	if (!datatype) datatype = 'float';
	var addon = utils.getAddon(datatype);
	return new Promise(function(resolve, reject) {
		var fannyTrainingData = new addon.TrainingData();
		fannyTrainingData.readTrainFromFile(filename, function(err) {
			if (err) return reject(new XError(err));
			var td = new TrainingData(fannyTrainingData, datatype);
			resolve(td);
		});
	});
}

module.exports = {
	createTrainingData: createTrainingData,
	loadTrainingData: loadTrainingData
};

