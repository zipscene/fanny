var createSchema = require('common-schema').createSchema;
var FieldError = require('common-schema').FieldError;
var XError = require('xerror');
var utils = require('./utils');
var createTrainingData = require('./training-data').createTrainingData;
var pasync = require('pasync');

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
		},
		activationFunctions: {
			type: 'map',
			values: {
				type: String,
				required: true,
				enum: [
					'LINEAR', 'THRESHOLD', 'THRESHOLD_SYMMETRIC', 'SIGMOID', 'SIGMOID_STEPWISE',
					'SIGMOID_SYMMETRIC', 'SIGMOID_SYMMETRIC_STEPWISE', 'GAUSSIAN', 'GAUSSIAN_SYMMETRIC',
					'ELLIOT', 'ELLIOT_SYMMETRIC', 'LINEAR_PIECE', 'LINEAR_PIECE_SYMMETRIC', 'SIN_SYMMETRIC',
					'COS_SYMMETRIC'
				]
			},
			validate: validateNeuronConfigKey
		},
		activationSteepnesses: {
			type: 'map',
			values: Number,
			validate: validateNeuronConfigKey
		}
	}
});

var annOptionsSchema = createSchema({
	trainingAlgorithm: {
		type: String,
		enum: [ 'INCREMENTAL', 'BATCH', 'RPROP', 'QUICKPROP', 'SARPROP' ]
	}
	// ADD MORE OPTIONS HERE
	// Note that, for enums, common prefixes (such as "FANN_" or "TRAIN_" should be removed
	// and converted in the getValue and setValue functions, as with the trainingAlgorithm option)
});

function validateNeuronConfigKey(val) {
	var keyRegex = /^[0-9]+$|^[0-9]+-[0-9]+$|^hidden$|^output$/;
	for (var key in val) {
		if (!keyRegex.test(key)) {
			var msg = 'Keys must be "hidden", "output", a layer number, ' +
				'or <Layer>-<Neuron> for a single neuron.';
			throw new FieldError('invalid', msg);
		}
	}
}


function wrapThrows(fn) {
	return function() {
		try {
			return fn.apply(this, Array.prototype.slice.call(arguments, 0));
		} catch (ex) {
			if (!XError.isXError(ex)) {
				throw new XError(ex);
			} else {
				throw ex;
			}
		}
	};
}

function nextQueueOp(ann) {
	if (ann._opQueue.length) {
		ann._currentlyRunning = true;
		var op = ann._opQueue.shift();
		var promise;
		try {
			promise = op.fn.apply(ann, op.args);
		} catch (ex) {
			op.waiter.reject(ex);
			nextQueueOp(ann);
			return;
		}
		if (!promise || typeof promise.then !== 'function') {
			op.waiter.resolve(promise);
			nextQueueOp(ann);
			return;
		}
		promise.then(function(res) {
			op.waiter.resolve(res);
			nextQueueOp(ann);
		}, function(err) {
			op.waiter.reject(err);
			nextQueueOp(ann);
		}).catch(pasync.abort);
	} else {
		ann._currentlyRunning = false;
	}
}

function asyncOpQueue(fn) {
	fn = wrapThrows(fn);
	return function() {
		var self = this;
		var waiter = pasync.waiter();
		self._opQueue.push({
			fn: fn,
			args: Array.prototype.slice.call(arguments, 0),
			waiter: waiter
		});
		if (!self._currentlyRunning) nextQueueOp(self);
		return waiter.promise;
	};
};

function blockOnAsync(fn) {
	return function() {
		if (this._currentlyRunning) {
			throw new XError(XError.INTERNAL_ERROR, 'Cannot execute this operation while training or running ann');
		}
		return fn.apply(this, Array.prototype.slice.call(arguments, 0));
	};
};

function ANN(fanny, datatype) {
	this._fanny = fanny;
	this._datatype = datatype;
	this._recalculateInfo();
	this._opQueue = [];
	this._currentlyRunning = false;
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
	/* Other options:
	- learningRate
	- trainErrorFunction
	- quickpropDecay
	- quickpropMu
	- rpropIncreaseFactor
	- rpropDecreaseFactor
	- rpropDeltaZero
	- rpropDeltaMin
	- rpropDeltaMax
	- sarpropWeightDecayShift
	- sarpropStepErrorThresholdFactor
	- sarpropStepErrorShift
	- sarpropTemperature
	- learningMomentum
	- trainStopFunction
	- bitFailLimit
	- cascadeOutputChangeFraction
	- cascadeOutputStagnationEpochs
	- cascadeCandidateChangeFraction
	- cascadeCandidateStagnationEpochs
	- cascadeWeightMultiplier
	- cascadeCandidateLimit
	- cascadeMaxOutEpochs
	- cascadeMaxCandEpochs
	- cascadeActivationFunctions
	- cascadeActivationSteepnesses
	- cascadeNumCandidateGroups
	*/
};

ANN.prototype.setOptions = blockOnAsync(function(options) {
	annOptionsSchema.normalize(options);
	for (var key in options) {
		if (!annOptionFunctions[key] || !annOptionFunctions[key].setValue) throw new XError(XError.INTERNAL_ERROR, 'No setter for option ' + key);
		annOptionFunctions[key].setValue.call(this, value);
	}
	this._recalculateInfo();
});

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
	var ret = {};
	for (var key in annOptionFunctions) {
		ret[key] = annOptionFunctions.getValue.call(this);
	}
	return ret;
};

ANN.prototype.clone = blockOnAsync(function() {
	var addon = utils.getAddon(this._datatype);
	var fanny = new addon.FANNY(this._fanny);
	return new ANN(fanny, this._datatype);
});

ANN.prototype._recalculateInfo = function() {
	this.info = {};
	var fns = {
		numInput: 'getNumInput',
		numOutput: 'getNumOutput',
		totalNeurons: 'getTotalNeurons',
		totalConnections: 'getTotalConnections',
		decimalPoint: 'getDecimalPoint',
		multiplier: 'getMultiplier',
		networkType: 'getNetworkType',
		connectionRate: 'getConnectionRate',
		numLayers: 'getNumLayers'
	};
	for (var key in fns) {
		if (this._fanny[fns[key]]) {
			this.info[key] = this._fanny[fns[key]]();
		}
	}
};

ANN.prototype.save = asyncOpQueue(function(filename, toFixed) {
	var self = this;
	return new Promise(function(resolve, reject) {
		var cb = function(err) {
			if (err) return reject(err);
			resolve();
		};
		if (toFixed) {
			self._fanny.saveToFixed(filename, cb);
		} else {
			self._fanny.save(filename, cb);
		}
	});
});

// data can either be a TrainingData class or a filename
// options can include: maxEpochs, progressInterval (in epochs), desiredError, cascade (boolean true for cascade training),
//   maxNeurons (for cascade training), stopFunction (either "MSE" or "BIT").  Without supplying an options object, this
//   only trains a single epoch.
// progress is an optional callback that is periodically called for multi-epoch training.  It receives a single
//   parameter: an object containing the keys "iteration" (current epoch or neuron), "mse", and "bitfail".  If
//   this progress function returns false or -1, training is cancelled on the next iteration.
//   Instead of a function, you can instead pass the special value "default", to enable the default libfann
//   behavior of printing out progress information.
ANN.prototype.train = asyncOpQueue(function(data, options, progress) {
	if (Array.isArray(data) && Array.isArray(options)) return this.trainOne(data, options);
	var self = this;
	var filename;
	var addonTrainingData;
	if (Array.isArray(data)) data = createTrainingData(data);
	if (data && typeof data === 'object' && typeof data.setTrainData === 'function') {
		addonTrainingData = data;
	} else if (data && typeof data === 'object' && typeof data.setData === 'function') {
		addonTrainingData = data._fannyTrainingData;
	} else if (typeof data === 'string') {
		filename = data;
	} else {
		throw new XError(XError.INVALID_ARGUMENT, 'Invalid training data type');
	}
	if (!options) {
		if (!addonTrainingData) throw new XError(XError.INVALID_ARGUMENT, 'Cannot train single epoch from file');
		return new Promise(function(resolve, reject) {
			self._fanny.trainEpoch(addonTrainingData, function(err, res) {
				if (err) return reject(new XError(err));
				resolve(res);
			});
		});
	}
	if (!options.maxEpochs && !options.maxNeurons && !options.desiredError) options.desiredError = 0.01;
	if (!options.maxEpochs && !options.maxNeurons) options.maxEpochs = 2000000000;
	if (!options.desiredError) options.desiredError = 0;
	if (!options.progressInterval) options.progressInterval = 1;
	if (options.stopFunction) {
		self._fanny.set_train_stop_function('STOPFUNC_' + options.stopFunction);
	}
	if (progress === 'default') {
		self._fanny.setCallback();
	} else if (typeof progress === 'function') {
		self._fanny.setCallback(function(info) {
			var result = progress(info);
			if (result === false || result === -1) return -1;
		});
	} else {
		self._fanny.setCallback(function() {});
	}
	return new Promise(function(resolve, reject) {
		var cb = function(err, res) {
			if (err) return reject(new XError(err));
			resolve(res);
		};
		var args = [ addonTrainingData || filename, options.maxEpochs || options.maxNeurons, options.progressInterval, options.desiredError, cb ];
		if (!options.cascade) {
			if (filename) {
				self._fanny.trainOnFile.apply(self._fanny, args);
			} else {
				self._fanny.trainOnData.apply(self._fanny, args);
			}
		} else {
			if (filename) {
				self._fanny.cascadetrainOnFile.apply(self._fanny, args);
			} else {
				self._fanny.cascadetrainOnData.apply(self._fanny, args);
			}
		}
	});
});

ANN.prototype.run = blockOnAsync(function(inputs) {
	return this._fanny.run(inputs);
});

ANN.prototype.runAsync = asyncOpQueue(function(inputs) {
	var self = this;
	return new Promise(function(resolve, reject) {
		self._fanny.runAsync(inputs, function(err, res) {
			if (err) return reject(new XError(err));
			resolve(res);
		});
	});
});

/* NEED TO ADD THESE PROTOTYPE METHODS: (a few should be renamed in JS, the C++ code should still expose the original names)
- randomizeWeights
- initWeights
- printConnections
- train (rename to trainOne)
- getMSE
- resetMSE
- printParameters
- getActivationFunction
- setActivationFunction
- setActivationFunctionLayer
- setActivationFunctionHidden
- setActivationFunctionOutput
- getActivationSteepness
- setActivationSteepness
- setActivationSteepnessLayer
- setActivationSteepnessHidden
- setActivationSteepnessOutput
- getLayerArray
- getBiasArray
- getConnectionArray
- setWeightArray
- setWeight
- scaleTrain (rename to scaleTrainingData)
- descaleTrain (rename to descaleTrainingData)
- setInputScalingParams
- setOutputScalingParams
- setScalingParams
- clearScalingParams
- scaleInput
- scaleOutput
- descaleInput
- descaleOutput
- test (rename to testOne)
- testData
*/

for (var key in ANN.prototype) {
	ANN.prototype[key] = wrapThrows(ANN.prototype[key]);
}

function createANN(config, options) {
	if (!config) throw new XError(XError.INVALID_ARGUMENT, 'config is required');
	if (Array.isArray(config)) config = { layers: config };
	annConfigSchema.normalize(config);
	var addon = utils.getAddon(config.datatype);
	var fanny = new addon.FANNY(config);
	if (config.activationFunctions) {
		for (var key in config.activationFunctions) {
			var value = 'FANN_' + config.activationFunctions[key];
			if (key === 'hidden') {
				fanny.setActivationFunctionHidden(value);
			} else if (key === 'output') {
				fanny.setActivationFunctionOutput(value);
			} else if (/^[0-9]+$/.test(key)) {
				fanny.setActivationFunctionLayer(value, parseInt(key));
			} else {
				var matches = /^([0-9]+)-([0-9]+)$/.exec(key);
				if (matches) {
					fanny.setActivationFunction(value, parseInt(matches[1]), parseInt(matches[2]));
				}
			}
		}
	}
	if (config.activationSteepnesses) {
		for (var key in config.activationSteepnesses) {
			var value = 'FANN_' + config.activationSteepnesses[key];
			if (key === 'hidden') {
				fanny.setActivationSteepnessHidden(value);
			} else if (key === 'output') {
				fanny.setActivationSteepnessOutput(value);
			} else if (/^[0-9]+$/.test(key)) {
				fanny.setActivationSteepnessLayer(value, parseInt(key));
			} else {
				var matches = /^([0-9]+)-([0-9]+)$/.exec(key);
				if (matches) {
					fanny.setActivationSteepness(value, parseInt(matches[1]), parseInt(matches[2]));
				}
			}
		}
	}
	var ann = new ANN(fanny, config.datatype);
	// FANN seeds the libc PRNG every time a neural net is created.  We want to disable this after the
	// first time it's seeded.
	//addon.FANNY.disableSeedRand(); // UNCOMMENT THIS WHEN FUNCTION IS ADDED
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

