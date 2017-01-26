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
	// - bitFailLimit
	// - cascadeOutputChangeFraction
	// - cascadeOutputStagnationEpochs
	// - cascadeCandidateChangeFraction
	// - cascadeCandidateStagnationEpochs
	// - cascadeWeightMultiplier
	// - cascadeCandidateLimit
	// - cascadeMaxOutEpochs
	// - cascadeMaxCandEpochs
	// - cascadeActivationFunctions
	// - cascadeActivationSteepnesses
	// - cascadeNumCandidateGroups
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


function ANN(fanny, datatype) {
	this._fanny = fanny;
	this._datatype = datatype;
	this._recalculateInfo();
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
	},

	learningRate: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid learningRate type');
			this._fanny.setLearningRate(value);
		},
		getValue: function() {
			return this._fanny.getLearningRate();
		}
	},

	trainErrorFunction: {
		setValue: function(value) {
			if ((typeof value !== 'string')) throw new XError(XError.INVALID_ARGUMENT, 'Invalid trainErrorFunction type');
			var trainErrorFunction = value.includes('ERRORFUNC_') ? value : 'ERRORFUNC_' + value;
			this._fanny.setTrainErrorFunction(trainErrorFunction);
		},
		getValue: function() {
			var trainErrorFunction = this._fanny.getTrainErrorFunction();
			if (trainErrorFunction.slice(0, 9) === 'ERRORFUNC_') return trainErrorFunction.slice(9);
			return trainErrorFunction;
		}
	},

	quickpropDecay: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid quickpropDecay type');
			this._fanny.setQuickpropDecay(value);
		},
		getValue: function() {
			return this._fanny.getQuickpropDecay();
		}
	},

	quickpropMu: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid quickpropMu type');
			if (value <= 1) throw new XError(XError.INVALID_ARGUMENT, 'quickpropMu must be > 1');
			this._fanny.setQuickpropMu(value);
		},
		getValue: function() {
			return this._fanny.getQuickpropMu();
		}
	},

	rpropIncreaseFactor: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid rpropIncreaseFactor type');
			if (value <= 1) throw new XError(XError.INVALID_ARGUMENT, 'rpropIncreaseFactor must be > 1');
			this._fanny.setRpropIncreaseFactor(value);
		},
		getValue: function() {
			return this._fanny.getRpropIncreaseFactor();
		}
	},

	rpropDecreaseFactor: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid rpropDecreaseFactor type');
			if (value >= 1) throw new XError(XError.INVALID_ARGUMENT, 'rpropDecreaseFactor must be < 1');
			this._fanny.setRpropDecreaseFactor(value);
		},
		getValue: function() {
			return this._fanny.getRpropDecreaseFactor();
		}
	},

	rpropDeltaZero: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid rpropDeltaZero type');
			if (value < 0) throw new XError(XError.INVALID_ARGUMENT, 'rpropDeltaZero must be > 0');
			this._fanny.setRpropDeltaZero(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaZero();
		}
	},

	rpropDeltaMax: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid rpropDeltaMax type');
			if (value < 0) throw new XError(XError.INVALID_ARGUMENT, 'rpropDeltaMax must be > 0');
			this._fanny.setRpropDeltaMax(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaMax();
		}
	},

	rpropDeltaMin: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid rpropDeltaMax type');
			if (value < 0) throw new XError(XError.INVALID_ARGUMENT, 'rpropDeltaMin must be > 0');
			this._fanny.setRpropDeltaMax(value);
		},
		getValue: function() {
			return this._fanny.getRpropDeltaMax();
		}
	},

	sarpropWeightDecayShift: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid sarpropWeightDecayShift type');
			this._fanny.setSarpropWeightDecayShift(value);
		},
		getValue: function() {
			return this._fanny.getSarpropWeightDecayShift();
		}
	},

	sarpropStepErrorThresholdFactor: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid sarpropStepErrorThresholdFactor type');
			this._fanny.setSarpropStepErrorThresholdFactor(value);
		},
		getValue: function() {
			return this._fanny.getSarpropStepErrorThresholdFactor();
		}
	},

	sarpropStepErrorShift: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid sarpropStepErrorShift type');
			this._fanny.setSarpropStepErrorShift(value);
		},
		getValue: function() {
			return this._fanny.getSarpropStepErrorShift();
		}
	},

	sarpropTemperature: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid sarpropTemperature type');
			this._fanny.setSarpropTemperature(value);
		},
		getValue: function() {
			return this._fanny.getSarpropTemperature();
		}
	},

	learningMomentum: {
		setValue: function(value) {
			if ((typeof value !== 'number') || Number.isNaN(value)) throw new XError(XError.INVALID_ARGUMENT, 'Invalid learningMomentum type');:
// TODO: http://libfann.github.io/fann/docs/files/fann_cpp-h.html#neural_net.get_learning_rate recommend this value be between 0.0 and 1.0 but doesn't enforce that do we want to.

			this._fanny.setLearningMomentum(value);
		},
		getValue: function() {
			return this._fanny.getLearningMomentum();
		}
	},

	trainStopFunction: {
			setValue: function(value) {
				if ((typeof value !== 'string')) throw new XError(XError.INVALID_ARGUMENT, 'Invalid trainStopFunction type');
				var trainStopFunction = value.includes('STOPFUNC_') ? value : 'STOPFUNC_' + value;
				this._fanny.setTraiStopFunction(trainStopFunction);
			},
			getValue: function() {
				var trainStopFunction = this._fanny.getTrainStopFunction();
				if (trainStopFunction.slice(0, 8) === 'STOPFUNC_') return trainStopFunction.slice(8);
				return trainStopFunction;
			}
		},
	// ADD MORE OPTIONS HERE
	/* Other options:
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

ANN.prototype.setOptions = function(options) {
	annOptionsSchema.normalize(options);
	for (var key in options) {
		if (!annOptionFunctions[key] || !annOptionFunctions[key].setValue) throw new XError(XError.INTERNAL_ERROR, 'No setter for option ' + key);
		annOptionFunctions[key].setValue.call(this, value);
	}
	this._recalculateInfo();
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
	var ret = {};
	for (var key in annOptionFunctions) {
		ret[key] = annOptionFunctions.getValue.call(this);
	}
	return ret;
};

ANN.prototype.clone = function() {
	var addon = utils.getAddon(this._datatype);
	var fanny = new addon.FANNY(this._fanny);
	return new ANN(fanny, this._datatype);
};

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

ANN.prototype.save = function(filename, toFixed) {
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
};

// data can either be a TrainingData class or a filename
// options can include: maxEpochs, progressInterval (in epochs), desiredError, cascade (boolean true for cascade training),
//   maxNeurons (for cascade training), stopFunction (either "MSE" or "BIT").  Without supplying an options object, this
//   only trains a single epoch.
// progress is an optional callback that is periodically called for multi-epoch training.  It receives a single
//   parameter: an object containing the keys "iteration" (current epoch or neuron), "mse", and "bitfail".  If
//   this progress function returns false or -1, training is cancelled on the next iteration.
//   Instead of a function, you can instead pass the special value "default", to enable the default libfann
//   behavior of printing out progress information.
ANN.prototype.train = function(data, options, progress) {
	if (Array.isArray(data) && Array.isArray(options)) return this.trainOne(data, options);
	var self = this;
	var filename;
	var addonTrainingData;
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
};

ANN.prototype.run = function(inputs) {
};

ANN.prototype.runAsync = function(inputs) {

};

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
