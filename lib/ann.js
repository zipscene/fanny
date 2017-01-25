var createSchema = require('common-schema').createSchema;
var FieldError = require('common-schema').FieldError;
var XError = require('xerror');
var utils = require('./utils');

var annConfigSchema = createSchema({
	type: 'object',
	properties: {
		type: {
			type: String,
			default: 'standard',
			enum: [ 'standard', 'sparse', 'shortcut' ]
		},
		layers: {
			type: [ {
				type: Number,
				required: true,
				min: 1
			} ],
			required: true,
			validate: function(val) {
				if (val.length < 2) throw new FieldError('invalid', 'Must have at least 2 layers');
			}
		},
		connectionRate: {
			type: Number,
			min: 0,
			max: 1
		},
		datatype: {
			type: String,
			default: 'float',
			enum: [ 'float', 'double', 'fixed' ]
		}
	}
});

var annOptionsSchema = createSchema({
	trainingAlgorithm: {
		type: String,
		enum: [ 'INCREMENTAL', 'BATCH', 'RPROP', 'QUICKPROP', 'SARPROP' ]
	}
	// ADD MORE OPTIONS HERE
});

function ANN(fanny, datatype) {
	this._fanny = fanny;
	this._datatype = datatype;
}

var annOptionFunctions = {
	trainingAlgorithm: {
		setValue: function(value) {
			var fannTrainingAlgorithmName = (options.trainingAlgorithm === 'SARPROP') ? 'FANN_TRAIN_SARPROP' : ('TRAIN_' + options.trainingAlgorithm);
			this._fanny.setTrainingAlgorithm(fannTrainingAlgorithmName);
		},
		getValue: function() {
			var fannTrainingAlgorithmName = this._fanny.getTrainingAlgorithm();
			if (fannTrainingAlgorithmName === 'FANN_TRAIN_SARPROP') return 'SARPROP';
			if (fannTrainingAlgorithmName.slice(0, 6) === 'TRAIN_') return fannTrainingAlgorithmName.slice(6);
			return fannTrainingAlgorithmName;
		}
	}
	// ADD MORE OPTIONS HERE
};

ANN.prototype.setOptions = function(options) {
	annOptionsSchema.normalize(options);
	for (let key in options) {
		if (!annOptionFunctions[key] || !annOptionFunctions[key].setValue) throw new XError(XError.INTERNAL_ERROR, 'No setter for option ' + key);
		annOptionFunctions[key].setValue.call(this, value);
	}
};

ANN.prototype.getOption = function(name) {
	if (!annOptionFunctions[key] || !annOptionFunctions[key].getValue) throw new XError(XError.INTERNAL_ERROR, 'No getter for option ' + name);
	return annOptionFunctions[key].getValue.call(this);
};

ANN.prototype.setOption = function(name, value) {
	var obj = {};
	obj.name = value;
	return this.setOptions(obj);
};

ANN.prototype.getOptions = function() {
	let ret = {};
	for (let key in annOptionFunctions) {
		ret[key] = annOptionFunctions.getValue.call(this);
	}
	return ret;
};

ANN.prototype.clone = function() {
	var addon = utils.getAddon(this._datatype);
	var fanny = new addon.FANNY(this._fanny);
	return new ANN(fanny, this._datatype);
};

function createANN(config, options) {
	if (!config) throw new XError(XError.INVALID_ARGUMENT, 'config is required');
	if (Array.isArray(config)) config = { layers: config };
	annConfigSchema.normalize(config);
	var addon = utils.getAddon(config.datatype);
	var fanny = new addon.FANNY(config);
	var ann = new ANN(fanny, config.datatype);
	if (options) ann.setOptions(options);
	return ann;
}

function loadANN(filename, datatype) {
	if (!filename) throw new XError(XError.INVALID_ARGUMENT, 'filename is required');
	if (!datatype) datatype = 'float';
	var addon = utils.getAddon(datatype);
	return new Promise(function(resolve, reject) {
		addon.FANNY.loadFile(filename, function(err, fanny) {
			if (err) return reject(new XError(err));
			var ann = new ANN(fanny, datatype);
			resolve(ann);
		});
	});
}

module.exports = {
	createANN: createANN,
	loadANN: loadANN
};

