// Copyright 2016 Zipscene, LLC
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

var createSchema = require('common-schema').createSchema;
var FieldError = require('common-schema').FieldError;
var XError = require('xerror');
var utils = require('./utils');
var createTrainingData = require('./training-data').createTrainingData;
var pasync = require('pasync');
var ACTIVATION_FUNCTIONS = [
	'LINEAR', 'THRESHOLD', 'THRESHOLD_SYMMETRIC', 'SIGMOID', 'SIGMOID_STEPWISE',
	'SIGMOID_SYMMETRIC', 'SIGMOID_SYMMETRIC_STEPWISE', 'GAUSSIAN', 'GAUSSIAN_SYMMETRIC',
	'ELLIOT', 'ELLIOT_SYMMETRIC', 'LINEAR_PIECE', 'LINEAR_PIECE_SYMMETRIC', 'SIN_SYMMETRIC',
	'COS_SYMMETRIC', 'SIN', 'COS'
];

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
				enum: ACTIVATION_FUNCTIONS
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

// Note that, for enums, common prefixes (such as "FANN_" or "TRAIN_" are removed
// and converted in the getValue and setValue functions, as with the trainingAlgorithm option)
var annOptionsSchema = createSchema({
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
		type: Number,
		min: 1
	},
	rpropIncreaseFactor: {
		type: Number,
		min: 1
	},
	rpropDecreaseFactor: {
		type: Number,
		max: 1	},
	rpropDeltaZero: {
		type: Number,
		min: 0
	},
	rpropDeltaMin: {
		type: Number,
		min: 0
	},
	rpropDeltaMax: {
		type: Number,
		min: 0
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
		type: Number,
		min: 0,
		max: 1
	},
	cascadeOutputStagnationEpochs: {
		type: Number
	},
	cascadeCandidateChangeFraction: {
		type: Number,
		min: 0,
		max: 1
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
			enum: ACTIVATION_FUNCTIONS
		}
	},
	cascadeActivationSteepnesses: {
		type: 'array',
		elements: {
			type: 'number',
			required: true
		}
	},
	cascadeNumCandidateGroups: {
		type: Number
	},
	userDataString: {
		type: String
	}
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
	this.userData = {};
	this._fanny = fanny;
	this._datatype = datatype;
	this._recalculateInfo();
	this._opQueue = [];
	this._currentlyRunning = false;
	var userDataString = this.getOption('userDataString');
	if (userDataString && userDataString[0] === '{') {
		this.userData = JSON.parse(userDataString);
	}
}

var annOptionFunctions = {
	trainingAlgorithm: {
		setValue: function(value) {
			var fannTrainingAlgorithmName = (value === 'SARPROP') ? 'FANN_TRAIN_SARPROP' : ('TRAIN_' + value);
			this._fanny.setTrainingAlgorithm(fannTrainingAlgorithmName);
		},
		getValue: function() {
			var fannTrainingAlgorithmName = this._fanny.getTrainingAlgorithm();
			if (fannTrainingAlgorithmName === 'FANN_TRAIN_SARPROP') return 'SARPROP';
			if (fannTrainingAlgorithmName.slice(0, 6) === 'TRAIN_') return fannTrainingAlgorithmName.slice(6);
			return fannTrainingAlgorithmName;
		}
	},

	learningRate: {
		setValue: function(value) {
			this._fanny.setLearningRate(value);
		},
		getValue: function() {
			return this._fanny.getLearningRate();
		}
	},

	trainErrorFunction: {
		setValue: function(value) {
			this._fanny.setTrainErrorFunction('ERRORFUNC_' + value);
		},
		getValue: function() {
			var trainErrorFunction = this._fanny.getTrainErrorFunction();
			if (trainErrorFunction.slice(0, 10) === 'ERRORFUNC_') return trainErrorFunction.slice(10);
			return trainErrorFunction;
		}
	},

	quickpropDecay: {
		setValue: function(value) {
			this._fanny.setQuickpropDecay(value);
		},
		getValue: function() {
			return this._fanny.getQuickpropDecay();
		}
	},

	quickpropMu: {
		setValue: function(value) {
			this._fanny.setQuickpropMu(value);
		},
		getValue: function() {
			return this._fanny.getQuickpropMu();
		}
	},

	rpropIncreaseFactor: {
		setValue: function(value) {
			this._fanny.setRpropIncreaseFactor(value);
		},
		getValue: function() {
			return this._fanny.getRpropIncreaseFactor();
		}
	},

	rpropDecreaseFactor: {
		setValue: function(value) {
			this._fanny.setRpropDecreaseFactor(value);
		},
		getValue: function() {
			return this._fanny.getRpropDecreaseFactor();
		}
	},

	rpropDeltaZero: {
		setValue: function(value) {
			this._fanny.setRpropDeltaZero(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaZero();
		}
	},

	rpropDeltaMax: {
		setValue: function(value) {
			this._fanny.setRpropDeltaMax(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaMax();
		}
	},

	rpropDeltaMin: {
		setValue: function(value) {
			this._fanny.setRpropDeltaMax(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaMax();
		}
	},

	sarpropWeightDecayShift: {
		setValue: function(value) {
			this._fanny.setSarpropWeightDecayShift(value);
		},
		getValue: function() {
			return this._fanny.getSarpropWeightDecayShift();
		}
	},

	sarpropStepErrorThresholdFactor: {
		setValue: function(value) {
			this._fanny.setSarpropStepErrorThresholdFactor(value);
		},
		getValue: function() {
			return this._fanny.getSarpropStepErrorThresholdFactor();
		}
	},

	sarpropStepErrorShift: {
		setValue: function(value) {
			this._fanny.setSarpropStepErrorShift(value);
		},
		getValue: function() {
			return this._fanny.getSarpropStepErrorShift();
		}
	},

	sarpropTemperature: {
		setValue: function(value) {
			this._fanny.setSarpropTemperature(value);
		},
		getValue: function() {
			return this._fanny.getSarpropTemperature();
		}
	},

	learningMomentum: {
		setValue: function(value) {
			this._fanny.setLearningMomentum(value);
		},
		getValue: function() {
			return this._fanny.getLearningMomentum();
		}
	},

	trainStopFunction: {
		setValue: function(value) {
			this._fanny.setTrainStopFunction('STOPFUNC_' + value);
		},
		getValue: function() {
			var trainStopFunction = this._fanny.getTrainStopFunction();
			if (trainStopFunction.slice(0, 9) === 'STOPFUNC_') return trainStopFunction.slice(9);
			return trainStopFunction;
		}
	},

	bitFailLimit: {
		setValue: function(value) {
			this._fanny.setBitFailLimit(value);
		},
		getValue: function() {
			return this._fanny.getBitFailLimit();
		}
	},

	cascadeOutputChangeFraction: {
		setValue: function(value) {
			this._fanny.setCascadeOutputChangeFraction(value);
		},
		getValue: function() {
			return this._fanny.getCascadeOutputChangeFraction();
		}
	},

	cascadeOutputStagnationEpochs: {
		setValue: function(value) {
			this._fanny.setCascadeOutputStagnationEpochs(value);
		},
		getValue: function() {
			return this._fanny.getCascadeOutputStagnationEpochs();
		}
	},

	cascadeCandidateChangeFraction: {
		setValue: function(value) {
			this._fanny.setCascadeCandidateChangeFraction(value);
		},
		getValue: function() {
			return this._fanny.getCascadeCandidateChangeFraction();
		}
	},

	cascadeCandidateStagnationEpochs: {
		setValue: function(value) {
			this._fanny.setCascadeCandidateStagnationEpochs(value);
		},
		getValue: function() {
			return this._fanny.getCascadeCandidateStagnationEpochs();
		}
	},

	cascadeWeightMultiplier: {
		setValue: function(value) {
			this._fanny.setCascadeWeightMultiplier(value);
		},
		getValue: function() {
			return this._fanny.getCascadeWeightMultiplier();
		}
	},

	cascadeCandidateLimit: {
		setValue: function(value) {
			this._fanny.setCascadeCandidateLimit(value);
		},
		getValue: function() {
			return this._fanny.getCascadeCandidateLimit();
		}
	},

	cascadeMaxOutEpochs: {
		setValue: function(value) {
			this._fanny.setCascadeMaxOutEpochs(value);
		},
		getValue: function() {
			return this._fanny.getCascadeMaxOutEpochs();
		}
	},

	cascadeMaxCandEpochs: {
		setValue: function(value) {
			this._fanny.setCascadeMaxCandEpochs(value);
		},
		getValue: function() {
			return this._fanny.getCascadeMaxCandEpochs();
		}
	},

	cascadeActivationFunctions: {
		setValue: function(value) {
			this._fanny.setCascadeActivationFunctions(value, value.length);
		},
		getValue: function() {
			return this._fanny.getCascadeActivationFunctions();
		}
	},
	cascadeActivationSteepnesses: {
		setValue: function(value) {
			this._fanny.setCascadeActivationSteepnesses(value, value.length);
		},
		getValue: function() {
			return this._fanny.getCascadeActivationSteepnesses();
		}
	},
	cascadeNumCandidateGroups: {
		setValue: function(value) {
			this._fanny.setCascadeNumCandidateGroups(value);
		},
		getValue: function() {
			return this._fanny.getCascadeNumCandidateGroups();
		}
	},
	userDataString: {
		setValue: function(value) {
			this._fanny.setUserDataString(value);
			if (value && value[0] === '{') {
				this.userData = JSON.parse(value);
			} else {
				this.userData = {};
			}
		},
		getValue: function() {
			return this._fanny.getUserDataString();
		}
	}
};

ANN.prototype.setOptions = blockOnAsync(function(options) {
	annOptionsSchema.normalize(options);
	for (var key in options) {
		if (!annOptionFunctions[key] || !annOptionFunctions[key].setValue) throw new XError(XError.INTERNAL_ERROR, 'No setter for option ' + key);
		annOptionFunctions[key].setValue.call(this, options[key]);
	}
	this._recalculateInfo();
});

ANN.prototype.getOption = function(name) {
	if (!annOptionFunctions[name] || !annOptionFunctions[name].getValue) throw new XError(XError.INTERNAL_ERROR, 'No getter for option ' + name);
	return annOptionFunctions[name].getValue.call(this);
};

ANN.prototype.setOption = function(name, value) {
	var obj = {};
	obj[name] = value;
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
		var curUserDataString = self.getOption('userDataString');
		if (!curUserDataString || curUserDataString[0] === '{') {
			self._fanny.setUserDataString(JSON.stringify(self.userData));
		}
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
	if (!options.maxEpochs && !options.maxNeurons && typeof options.desiredError !== 'number') options.desiredError = 0.01;
	if (!options.maxEpochs && !options.maxNeurons) options.maxEpochs = 2000000000;
	if (typeof options.desiredError !== 'number') options.desiredError = 0.01;
	if (!options.progressInterval) options.progressInterval = 1;
	if (options.stopFunction) {
		self._fanny.setTrainStopFunction('STOPFUNC_' + options.stopFunction);
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

ANN.prototype.randomizeWeights = blockOnAsync(wrapThrows(function(min, max) {
	if (typeof min !== 'number' || Number.isNaN(min)) throw new XError(XError.INVALID_ARGUMENT, 'min must be a number');
	if (typeof max !== 'number' || Number.isNaN(max)) throw new XError(XError.INVALID_ARGUMENT, 'max must be a number');
	return this._fanny.randomizeWeights(min, max);
}));

ANN.prototype.initWeights = blockOnAsync(wrapThrows(function(data) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instanceof TrianingData');
	}
	return this._fanny.initWeights(data._fannyTrainingData);
}));

ANN.prototype.printConnections = function() {
	return this._fanny.printConnections();
};

ANN.prototype.printParameters = function() {
	return this._fanny.printParameters();
};

ANN.prototype.getConnectionArray = function() {
	return this._fanny.getConnectionArray();
};

ANN.prototype.getBitFail = function() {
	return this._fanny.getBitFail();
};

ANN.prototype.getMSE = function() {
	return this._fanny.getMSE();
};

ANN.prototype.resetMSE = blockOnAsync(function() {
	return this._fanny.resetMSE();
});

ANN.prototype.getActivationFunction = wrapThrows(function(layer, neruon) {
	if (typeof layer !== 'number' || Number.isNaN(layer)) throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number');
	if (typeof neruon !== 'number' || Number.isNaN(neruon)) throw new XError(XError.INVALID_ARGUMENT, 'neruon must be a number');
	var activationFunction = this._fanny.getActivationFunction(layer, neruon);
	return activationFunction;
});

ANN.prototype.setActivationFunction = blockOnAsync(wrapThrows(function(activationFunction, layer, neruon) {
	if (typeof activationFunction !== 'string') throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be a string');
	if (!ACTIVATION_FUNCTIONS.find(function(name) { return name === activationFunction })) {
		throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be included in ' + ACTIVATION_FUNCTIONS.join(', '));
	}
	if (typeof layer !== 'number' || Number.isNaN(layer)) throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number');
	if (typeof neruon !== 'number' || Number.isNaN(neruon)) throw new XError(XError.INVALID_ARGUMENT, 'neruon must be a number');
	return this._fanny.setActivationFunction(activationFunction, layer, neruon);
}));

ANN.prototype.setActivationFunctionLayer = blockOnAsync(wrapThrows(function(activationFunction, layer) {
	// check activationFunction from enum
	if (typeof activationFunction !== 'string') throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be a string');
	if (!ACTIVATION_FUNCTIONS.find(function(name) { return name === activationFunction })) {
		throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be included in ' + ACTIVATION_FUNCTIONS.join(', '));
	}
	if (typeof layer !== 'number' || Number.isNaN(layer)) throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number');
	return this._fanny.setActivationFunctionLayer(activationFunction, layer);
}));

ANN.prototype.setActivationFunctionHidden = blockOnAsync(wrapThrows(function(activationFunction) {
	if (typeof activationFunction !== 'string') throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be a string');
	if (!ACTIVATION_FUNCTIONS.find(function(name) { return name === activationFunction })) {
		throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be included in ' + ACTIVATION_FUNCTIONS.join(', '));
	}
	return this._fanny.setActivationFunctionHidden(activationFunction);
}));

ANN.prototype.setActivationFunctionOutput = blockOnAsync(wrapThrows(function(activationFunction) {
	if (typeof activationFunction !== 'string') throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be a string');
	if (!ACTIVATION_FUNCTIONS.find(function(name) { return name === activationFunction })) {
		throw new XError(XError.INVALID_ARGUMENT, 'activationFunction must be included in ' + ACTIVATION_FUNCTIONS.join(', '));
	}
	return this._fanny.setActivationFunctionOutput(activationFunction);
}));

ANN.prototype.getLayerArray = function() {
	return this._fanny.getLayerArray();
};

ANN.prototype.getBiasArray = function() {
	return this._fanny.getBiasArray();
};

ANN.prototype.scaleTrainingData = blockOnAsync(wrapThrows(function(data) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instance of training data');
	}
	return this._fanny.scaleTrain(data._fannyTrainingData);
}));

ANN.prototype.descaleTrainingData = blockOnAsync(wrapThrows(function(data) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instance of training data');
	}
	return this._fanny.descaleTrain(data._fannyTrainingData);
}));

ANN.prototype.setInputScalingParams = blockOnAsync(wrapThrows(function(data, min, max) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instance of trainingData');
	}
	if (typeof min !== 'number' || Number.isNaN(min)) {
		throw new XError(XError.INVALID_ARGUMENT, 'min must be a number');
	}
	if (typeof max !== 'number'|| Number.isNaN(max)) {
		throw new XError(XError.INVALID_ARGUMENT, 'max must be a number');
	}
	return this._fanny.setInputScalingParams(data._fannyTrainingData, min, max);
}));

ANN.prototype.setOutputScalingParams = blockOnAsync(wrapThrows(function(data, min, max) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instance of trainingData');
	}
	if (typeof min !== 'number' || Number.isNaN(min)) {
		throw new XError(XError.INVALID_ARGUMENT, 'min must be a number');
	}
	if (typeof max !== 'number' || Number.isNaN(max)) {
		throw new XError(XError.INVALID_ARGUMENT, 'max must be a number');
	}
	return this._fanny.setOutputScalingParams(data._fannyTrainingData, min, max);
}));

ANN.prototype.setScalingParams = blockOnAsync(wrapThrows(function(data, inputMin, inputMax, outputMin, outputMax) {
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instance of trainingData');
	}
	if (typeof inputMin !== 'number' || Number.isNaN(inputMin)) {
		throw new XError(XError.INVALID_ARGUMENT, 'inputMin must be a number');
	}
	if (typeof inputMax !== 'number' || Number.isNaN(inputMax)) {
		throw new XError(XError.INVALID_ARGUMENT, 'inputMax must be a number');
	}
	if (typeof outputMin !== 'number' || Number.isNaN(outputMin)) {
		throw new XError(XError.INVALID_ARGUMENT, 'outputMin must be a number');
	}
	if (typeof outputMax !== 'number' || Number.isNaN(outputMax)) {
		throw new XError(XError.INVALID_ARGUMENT, 'outputMax must be a number');
	}
	return this._fanny.setScalingParams(data._fannyTrainingData, inputMin, inputMax, outputMin, outputMax);
}));

ANN.prototype.clearScalingParams = blockOnAsync(wrapThrows(function() {
	return this._fanny.clearScalingParams();
}));

ANN.prototype.scaleInput = blockOnAsync(wrapThrows(function(input) {
	if (!Array.isArray(input) || !input.length) {
		throw new XError(XError.INVALID_ARGUMENT, 'input must an array');
	}
	return this._fanny.scaleInput(input);
}));

ANN.prototype.scaleOutput = blockOnAsync(wrapThrows(function(output) {
	if (!Array.isArray(output) || !output.length) {
		throw new XError(XError.INVALID_ARGUMENT, 'output must an array');
	}
	return this._fanny.scaleOutput(output);
}));

ANN.prototype.descaleInput = blockOnAsync(wrapThrows(function(input) {
	if (!Array.isArray(input) || !input.length) {
		throw new XError(XError.INVALID_ARGUMENT, 'input must an array');
	}
	return this._fanny.descaleInput(input);
}));

ANN.prototype.descaleOutput = blockOnAsync(wrapThrows(function(output) {
	if (!Array.isArray(output) || !output.length) {
		throw new XError(XError.INVALID_ARGUMENT, 'output must an array');
	}
	return this._fanny.descaleOutput(output);
}));

ANN.prototype.getActivationSteepness = wrapThrows(function(layer, neuron) {
	if (typeof layer !== 'number' || Number.isNaN(layer)) {
		throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number');
	}
	if (typeof neuron !== 'number' || Number.isNaN(neuron)) {
		throw new XError(XError.INVALID_ARGUMENT, 'neuron must be a number');
	}
	return this._fanny.getActivationSteepness(layer, neuron);
});

ANN.prototype.setActivationSteepness = blockOnAsync(wrapThrows(function(steepness, layer, neuron) {
	if (typeof steepness !== 'number' || Number.isNaN(steepness)) {
		throw new XError(XError.INVALID_ARGUMENT, 'steepness must be a number');
	}
	if (typeof layer !== 'number' || Number.isNaN(layer) || layer < 0) {
		throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number and greater or equal to 0');
	}
	if (typeof neuron !== 'number' || Number.isNaN(neuron) || neuron < 0) {
		throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number and greater or equal to 0');
	}

	return this._fanny.setActivationSteepness(steepness, layer, neuron);
}));

ANN.prototype.setActivationSteepnessLayer = blockOnAsync(wrapThrows(function(steepness, layer) {
	if (typeof steepness !== 'number' || Number.isNaN(steepness)) {
		throw new XError(XError.INVALID_ARGUMENT, 'steepness must be a number');
	}
	if (typeof layer !== 'number' || Number.isNaN(layer) || layer < 0) {
		throw new XError(XError.INVALID_ARGUMENT, 'layer must be a number and greater or equal to 0');
	}
	return this._fanny.setActivationSteepnessLayer(steepness, layer);
}));

ANN.prototype.setActivationSteepnessHidden = blockOnAsync(wrapThrows(function(steepness) {
	if (typeof steepness !== 'number' || Number.isNaN(steepness)) {
		throw new XError(XError.INVALID_ARGUMENT, 'steepness must be a number');
	}
	return this._fanny.setActivationSteepnessHidden(steepness);
}));

ANN.prototype.setActivationSteepnessOutput = blockOnAsync(wrapThrows(function(steepness) {
	if (typeof steepness !== 'number' || Number.isNaN(steepness)) {
		throw new XError(XError.INVALID_ARGUMENT, 'steepness must be a number');
	}
	return this._fanny.setActivationSteepnessOutput(steepness);
}));

ANN.prototype.setWeightArray = blockOnAsync(wrapThrows(function(connections) {
	if (!Array.isArray(connections) || !connections.length) {
		throw new XError(XError.INVALID_ARGUMENT, 'connections must be an array');
	}

	for (var i = 0; i < connections.length; ++i) {
		if (!('toNeuron' in connections[i]) || !('fromNeuron' in connections[i]) || !('weight' in connections[i])) {
			throw new XError(XError.INVALID_ARGUMENT, 'all connections must have toNeuron, fromNeuron and weight');
		}
	}

	return this._fanny.setWeightArray(connections, connections.length);
}));

ANN.prototype.setWeight = blockOnAsync(wrapThrows(function(fromNeuron, toNeuron, weight) {
	if (typeof fromNeuron !== 'number' || Number.isNaN(fromNeuron) || fromNeuron < 0) {
		throw new XError(XError.INVALID_ARGUMENT, 'fromNeuron must be a number and greater or equal to 0');
	}
	if (typeof toNeuron !== 'number' || Number.isNaN(toNeuron) || toNeuron < 0) {
		throw new XError(XError.INVALID_ARGUMENT, 'toNeuron must be a number and greater or equal to 0');
	}
	if (typeof weight !== 'number' || Number.isNaN(weight)) {
		throw new XError(XError.INVALID_ARGUMENT, 'weight must be a number');
	}
	return this._fanny.setWeight(fromNeuron, toNeuron, weight);
}));

ANN.prototype.trainOne = blockOnAsync(wrapThrows(function(input, output) {
	if (!Array.isArray(input) || !Array.isArray(output)) {
		throw new XError(XError.INVALID_ARGUMENT, 'Both input and output should be arrays');
	}
	return this._fanny.train(input, output);
}));

ANN.prototype.testOne = blockOnAsync(wrapThrows(function(input, output) {
	if (!Array.isArray(input) || !Array.isArray(output)) {
		throw new XError(XError.INVALID_ARGUMENT, 'Both input and output should be arrays');
	}
	if (input.length !== this.info.numInput) {
		throw new XError(XError.INVALID_ARGUMENT, 'Input must be the right length');
	}
	if (output.length !== this.info.numOutput) {
		throw new XError(XError.INVALID_ARGUMENT, 'Output must be the right length');
	}
	return this._fanny.test(input, output);
}));

ANN.prototype.testData = asyncOpQueue(function(data) {
	var self = this;
	if (!data || !data._fannyTrainingData) {
		throw new XError(XError.INVALID_ARGUMENT, 'data must be an instanceof TrianingData');
	}
	return new Promise(function(resolve, reject) {
		self._fanny.testData(data._fannyTrainingData, function(err, res) {
			if (err) return reject(new XError(err));
			resolve(res);
		});
	});
});

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
			var value = config.activationFunctions[key];
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
			var value = config.activationSteepnesses[key];
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
	addon.FANNY.disableSeedRand();
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
	loadANN: loadANN,
	annConfigSchema: annConfigSchema,
	annOptionsSchema: annOptionsSchema
};
