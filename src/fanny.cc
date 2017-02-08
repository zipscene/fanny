#include "fanny.h"
#include <nan.h>
#include "fann-includes.h"
#include <iostream>
#include "utils.h"
#include "training-data.h"

namespace fanny {

class RunWorker : public Nan::AsyncWorker {

public:
	RunWorker(Nan::Callback *callback, std::vector<fann_type> & _inputs, v8::Local<v8::Object> fannyHolder) : Nan::AsyncWorker(callback), inputs(_inputs) {
		SaveToPersistent("fannyHolder", fannyHolder);
		fanny = Nan::ObjectWrap::Unwrap<FANNY>(fannyHolder);
	}
	~RunWorker() {}

	void Execute() {
		fann_type *fannOutputs = fanny->fann->run(&inputs[0]);
		if (fanny->fann->get_errno()) {
			SetErrorMessage(fanny->fann->get_errstr().c_str());
			fanny->fann->reset_errno();
			fanny->fann->reset_errstr();
			return;
		}
		unsigned int numOutputs = fanny->fann->get_num_output();
		for (unsigned int idx = 0; idx < numOutputs; idx++) {
			outputs.push_back(fannOutputs[idx]);
		}
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		v8::Local<v8::Value> args[] = {
			Nan::Null(),
			fannDataToV8Array(&outputs[0], outputs.size())
		};
		callback->Call(2, args);
	}

	std::vector<fann_type> inputs;
	std::vector<fann_type> outputs;
	FANNY *fanny;
};

class LoadFileWorker : public Nan::AsyncWorker {
public:
	LoadFileWorker(Nan::Callback *callback, std::string _filename) : Nan::AsyncWorker(callback), filename(_filename) {}
	~LoadFileWorker() {}

	void Execute() {
		struct fann *ann = fann_create_from_file(filename.c_str());
		if (!ann) return SetErrorMessage("Error loading FANN file");
		fann = new FANN::neural_net(ann);
		fann_destroy(ann);
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		if (fann->get_errno()) {
			v8::Local<v8::Value> args[] = { Nan::Error(std::string(fann->get_errstr()).c_str()) };
			delete fann;
			callback->Call(1, args);
		} else {
			v8::Local<v8::Value> externFann = Nan::New<v8::External>(fann);
			v8::Local<v8::Function> ctor = Nan::New(FANNY::constructorFunction);
			v8::Local<v8::Value> ctorArgs[] = { externFann };
			v8::Local<v8::Value> cbargs[] = {
				Nan::Null(),
				Nan::NewInstance(ctor, 1, ctorArgs).ToLocalChecked()
			};
			callback->Call(2, cbargs);
		}
	}

	std::string filename;
	FANN::neural_net *fann;
};

class SaveFileWorker : public Nan::AsyncWorker {
public:
	SaveFileWorker(Nan::Callback *callback, v8::Local<v8::Object> fannyHolder, std::string _filename, bool _isFixed) :
		Nan::AsyncWorker(callback), filename(_filename), isFixed(_isFixed)
	{
		SaveToPersistent("fannyHolder", fannyHolder);
		fanny = Nan::ObjectWrap::Unwrap<FANNY>(fannyHolder);
	}
	~SaveFileWorker() {}

	void Execute() {
		bool hasError = false;
		if (isFixed) {
			decimalPoint = fanny->fann->save_to_fixed(filename);
		} else {
			hasError = !fanny->fann->save(filename);
			decimalPoint = 0;
		}
		if (fanny->fann->get_errno()) {
			SetErrorMessage(fanny->fann->get_errstr().c_str());
			fanny->fann->reset_errno();
			fanny->fann->reset_errstr();
		} else if (hasError) {
			SetErrorMessage("Error saving FANN file");
		}
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		v8::Local<v8::Value> args[] = { Nan::Null(), Nan::New(decimalPoint) };
		callback->Call(2, args);
	}

	FANNY *fanny;
	std::string filename;
	bool isFixed;
	int decimalPoint;
};

class TrainWorker : public Nan::AsyncProgressWorker {
public:
	FANNY *fanny;
	TrainingData *trainingData;
	bool trainFromFile;
	std::string filename;
	bool isCascade;
	unsigned int maxIterations;
	unsigned int iterationsBetweenReports;
	float desiredError;
	bool singleEpoch;
	bool isTest;

	float retVal;
	const ExecutionProgress *executionProgress;

	TrainWorker(
		Nan::Callback *callback,
		v8::Local<v8::Object> fannyHolder,
		Nan::MaybeLocal<v8::Object> maybeTrainingDataHolder,
		bool _trainFromFile,
		std::string _filename,
		bool _isCascade,
		unsigned int _maxIterations,
		unsigned int _iterationsBetweenReports,
		float _desiredError,
		bool _singleEpoch,
		bool _isTest
	) : Nan::AsyncProgressWorker(callback), trainFromFile(_trainFromFile), filename(_filename),
	isCascade(_isCascade), maxIterations(_maxIterations), iterationsBetweenReports(_iterationsBetweenReports),
	desiredError(_desiredError), singleEpoch(_singleEpoch), isTest(_isTest), retVal(-1) {
		SaveToPersistent("fannyHolder", fannyHolder);
		fanny = Nan::ObjectWrap::Unwrap<FANNY>(fannyHolder);
		if (!maybeTrainingDataHolder.IsEmpty()) {
			v8::Local<v8::Object> trainingDataHolder = maybeTrainingDataHolder.ToLocalChecked();
			SaveToPersistent("tdHolder", trainingDataHolder);
			trainingData = Nan::ObjectWrap::Unwrap<TrainingData>(trainingDataHolder);
		}
	}

	void Execute(const ExecutionProgress &progress) {
		executionProgress = &progress;
		fanny->currentTrainWorker = this;
		fanny->cancelTrainingFlag = false;
		#ifndef FANNY_FIXED
		if (isTest) {
			retVal = fanny->fann->test_data(*trainingData->trainingData);
		} else if (singleEpoch) {
			retVal = fanny->fann->train_epoch(*trainingData->trainingData);
		} else if (!trainFromFile && !isCascade) {
			fanny->fann->train_on_data(*trainingData->trainingData, maxIterations, iterationsBetweenReports, desiredError);
		} else if (trainFromFile && !isCascade) {
			fanny->fann->train_on_file(filename, maxIterations, iterationsBetweenReports, desiredError);
		} else if (!trainFromFile && isCascade) {
			fanny->fann->cascadetrain_on_data(*trainingData->trainingData, maxIterations, iterationsBetweenReports, desiredError);
		} else if (trainFromFile && isCascade) {
			fanny->fann->cascadetrain_on_file(filename, maxIterations, iterationsBetweenReports, desiredError);
		}
		if (fanny->fann->get_errno()) {
			SetErrorMessage(fanny->fann->get_errstr().c_str());
			fanny->fann->reset_errno();
			fanny->fann->reset_errstr();
		} else if (!singleEpoch && !isTest) {
			retVal = fanny->fann->get_MSE();
		}
		#endif
		fanny->currentTrainWorker = NULL;
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		if (fanny->cancelTrainingFlag) {
			v8::Local<v8::Value> errorArgs[] = { Nan::Error("canceled") };
			callback->Call(1, errorArgs);
			return;
		}
		v8::Local<v8::Value> args[] = { Nan::Null(), Nan::New(retVal) };
		callback->Call(2, args);
	}

	void HandleProgressCallback(const char *_discard1, size_t _discard2) {
		Nan::HandleScope scope;
		if (!fanny->trainingCallbackFn.IsEmpty() && !fanny->cancelTrainingFlag) {
			v8::Local<v8::Function> trainingCallbackFn = Nan::New(fanny->trainingCallbackFn);
			v8::Local<v8::Object> obj = Nan::New<v8::Object>();
			Nan::Set(obj, Nan::New("iteration").ToLocalChecked(), Nan::New(fanny->currentTrainingProgress.iteration));
			Nan::Set(obj, Nan::New("mse").ToLocalChecked(), Nan::New(fanny->currentTrainingProgress.mse));
			Nan::Set(obj, Nan::New("bitfail").ToLocalChecked(), Nan::New(fanny->currentTrainingProgress.bitFail));
			v8::Local<v8::Value> args[] = { obj };
			Nan::MaybeLocal<v8::Value> ret = Nan::Call(trainingCallbackFn, GetFromPersistent("fannyHolder").As<v8::Object>(), 1, args);
			if (!ret.IsEmpty() && ret.ToLocalChecked()->IsNumber() && ret.ToLocalChecked()->Int32Value() < 0) {
				fanny->cancelTrainingFlag = true;
			}
		}
	}
};

void FANNY::Init(v8::Local<v8::Object> target) {
	// Create new function template for this JS class constructor
	v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
	// Set the class name
	tpl->SetClassName(Nan::New("FANNY").ToLocalChecked());
	// Set the number of "slots" to allocate for fields on this class, not including prototype methods
	tpl->InstanceTemplate()->SetInternalFieldCount(1);
	// Save a constructor reference
	FANNY::constructorFunctionTpl.Reset(tpl);

	// Add prototype methods

	Nan::SetPrototypeMethod(tpl, "printConnections", printConnections);
	Nan::SetPrototypeMethod(tpl, "printParameters", printParameters);
	Nan::SetPrototypeMethod(tpl, "randomizeWeights", randomizeWeights);

	Nan::SetPrototypeMethod(tpl, "save", save);
	Nan::SetPrototypeMethod(tpl, "saveToFixed", saveToFixed);
	Nan::SetPrototypeMethod(tpl, "setCallback", setCallback);
	Nan::SetPrototypeMethod(tpl, "trainEpoch", trainEpoch);
	Nan::SetPrototypeMethod(tpl, "trainOnData", trainOnData);
	Nan::SetPrototypeMethod(tpl, "trainOnFile", trainOnFile);
	Nan::SetPrototypeMethod(tpl, "cascadetrainOnData", cascadetrainOnData);
	Nan::SetPrototypeMethod(tpl, "cascadetrainOnFile", cascadetrainOnFile);
	Nan::SetPrototypeMethod(tpl, "run", run);
	Nan::SetPrototypeMethod(tpl, "getTrainingAlgorithm", getTrainingAlgorithm);
	Nan::SetPrototypeMethod(tpl, "setTrainingAlgorithm", setTrainingAlgorithm);
	Nan::SetPrototypeMethod(tpl, "getTrainErrorFunction", getTrainErrorFunction);
	Nan::SetPrototypeMethod(tpl, "setTrainErrorFunction", setTrainErrorFunction);
	Nan::SetPrototypeMethod(tpl, "getTrainStopFunction", getTrainStopFunction);
	Nan::SetPrototypeMethod(tpl, "setTrainStopFunction", setTrainStopFunction);
	Nan::SetPrototypeMethod(tpl, "getLearningRate", getLearningRate);
	Nan::SetPrototypeMethod(tpl, "setLearningRate", setLearningRate);
	Nan::SetPrototypeMethod(tpl, "getActivationFunction", getActivationFunction);
	Nan::SetPrototypeMethod(tpl, "setActivationFunction", setActivationFunction);
	Nan::SetPrototypeMethod(tpl, "setActivationFunctionLayer", setActivationFunctionLayer);
	Nan::SetPrototypeMethod(tpl, "setActivationFunctionHidden", setActivationFunctionHidden);
	Nan::SetPrototypeMethod(tpl, "setActivationFunctionOutput", setActivationFunctionOutput);
	Nan::SetPrototypeMethod(tpl, "getNumInput", getNumInput);
	Nan::SetPrototypeMethod(tpl, "getNumOutput", getNumOutput);
	Nan::SetPrototypeMethod(tpl, "getTotalNeurons", getTotalNeurons);
	Nan::SetPrototypeMethod(tpl, "getTotalConnections", getTotalConnections);
	Nan::SetPrototypeMethod(tpl, "getConnectionArray", getConnectionArray);
	Nan::SetPrototypeMethod(tpl, "getNumLayers", getNumLayers);
	Nan::SetPrototypeMethod(tpl, "getBitFail", getBitFail);
	Nan::SetPrototypeMethod(tpl, "getBitFailLimit", getBitFailLimit);
	Nan::SetPrototypeMethod(tpl, "setBitFailLimit", setBitFailLimit);
	Nan::SetPrototypeMethod(tpl, "getMSE", getMSE);
	Nan::SetPrototypeMethod(tpl, "resetMSE", resetMSE);
	Nan::SetPrototypeMethod(tpl, "getQuickpropDecay", getQuickpropDecay);
	Nan::SetPrototypeMethod(tpl, "getQuickpropMu", getQuickpropMu);
	Nan::SetPrototypeMethod(tpl, "getRpropIncreaseFactor", getRpropIncreaseFactor);
	Nan::SetPrototypeMethod(tpl, "getRpropDecreaseFactor", getRpropDecreaseFactor);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaZero", getRpropDeltaZero);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaMin", getRpropDeltaMin);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaMax", getRpropDeltaMax);
	Nan::SetPrototypeMethod(tpl, "runAsync", runAsync);
	Nan::SetPrototypeMethod(tpl, "initWeights", initWeights);
	Nan::SetPrototypeMethod(tpl, "testData", testData);
	Nan::SetPrototypeMethod(tpl, "getLayerArray", getLayerArray);
	Nan::SetPrototypeMethod(tpl, "getBiasArray", getBiasArray);
	Nan::SetPrototypeMethod(tpl, "train", train);
	Nan::SetPrototypeMethod(tpl, "test", test);
	Nan::SetPrototypeMethod(tpl, "scaleTrain", scaleTrain);
	Nan::SetPrototypeMethod(tpl, "descaleTrain", descaleTrain);
	Nan::SetPrototypeMethod(tpl, "clearScalingParams", clearScalingParams);
	Nan::SetPrototypeMethod(tpl, "setInputScalingParams", setInputScalingParams);
	Nan::SetPrototypeMethod(tpl, "setOutputScalingParams", setOutputScalingParams);
	Nan::SetPrototypeMethod(tpl, "setScalingParams", setScalingParams);
	Nan::SetPrototypeMethod(tpl, "scaleInput", scaleInput);
	Nan::SetPrototypeMethod(tpl, "scaleOutput", scaleOutput);
	Nan::SetPrototypeMethod(tpl, "descaleInput", descaleInput);
	Nan::SetPrototypeMethod(tpl, "descaleOutput", descaleOutput);

	Nan::SetPrototypeMethod(tpl, "getCascadeActivationFunctions", getCascadeActivationFunctions);
	Nan::SetPrototypeMethod(tpl, "setCascadeActivationFunctions", setCascadeActivationFunctions);
	Nan::SetPrototypeMethod(tpl, "getCascadeActivationSteepnesses", getCascadeActivationSteepnesses);
	Nan::SetPrototypeMethod(tpl, "setCascadeActivationSteepnesses", setCascadeActivationSteepnesses);

	Nan::SetPrototypeMethod(tpl, "getCascadeOutputChangeFraction", getCascadeOutputChangeFraction);
	Nan::SetPrototypeMethod(tpl, "getCascadeOutputStagnationEpochs", getCascadeOutputStagnationEpochs);
	Nan::SetPrototypeMethod(tpl, "getCascadeCandidateChangeFraction", getCascadeCandidateChangeFraction);
	Nan::SetPrototypeMethod(tpl, "getCascadeCandidateStagnationEpochs", getCascadeCandidateStagnationEpochs);
	Nan::SetPrototypeMethod(tpl, "getCascadeWeightMultiplier", getCascadeWeightMultiplier);
	Nan::SetPrototypeMethod(tpl, "getCascadeCandidateLimit", getCascadeCandidateLimit);
	Nan::SetPrototypeMethod(tpl, "getCascadeMaxOutEpochs", getCascadeMaxOutEpochs);
	Nan::SetPrototypeMethod(tpl, "getCascadeMaxCandEpochs", getCascadeMaxCandEpochs);
	Nan::SetPrototypeMethod(tpl, "getCascadeNumCandidateGroups", getCascadeNumCandidateGroups);

	Nan::SetPrototypeMethod(tpl, "setCascadeOutputChangeFraction", setCascadeOutputChangeFraction);
	Nan::SetPrototypeMethod(tpl, "setCascadeOutputStagnationEpochs", setCascadeOutputStagnationEpochs);
	Nan::SetPrototypeMethod(tpl, "setCascadeCandidateChangeFraction", setCascadeCandidateChangeFraction);
	Nan::SetPrototypeMethod(tpl, "setCascadeCandidateStagnationEpochs", setCascadeCandidateStagnationEpochs);
	Nan::SetPrototypeMethod(tpl, "setCascadeWeightMultiplier", setCascadeWeightMultiplier);
	Nan::SetPrototypeMethod(tpl, "setCascadeCandidateLimit", setCascadeCandidateLimit);
	Nan::SetPrototypeMethod(tpl, "setCascadeMaxOutEpochs", setCascadeMaxOutEpochs);
	Nan::SetPrototypeMethod(tpl, "setCascadeMaxCandEpochs", setCascadeMaxCandEpochs);
	Nan::SetPrototypeMethod(tpl, "setCascadeNumCandidateGroups", setCascadeNumCandidateGroups);

	Nan::SetPrototypeMethod(tpl, "setQuickpropDecay", setQuickpropDecay);
	Nan::SetPrototypeMethod(tpl, "setQuickpropMu", setQuickpropMu);
	Nan::SetPrototypeMethod(tpl, "setRpropIncreaseFactor", setRpropIncreaseFactor);
	Nan::SetPrototypeMethod(tpl, "setRpropDecreaseFactor", setRpropDecreaseFactor);
	Nan::SetPrototypeMethod(tpl, "setRpropDeltaZero", setRpropDeltaZero);
	Nan::SetPrototypeMethod(tpl, "setRpropDeltaMin", setRpropDeltaMin);
	Nan::SetPrototypeMethod(tpl, "setRpropDeltaMax", setRpropDeltaMax);

	Nan::SetPrototypeMethod(tpl, "getSarpropWeightDecayShift", getSarpropWeightDecayShift);
	Nan::SetPrototypeMethod(tpl, "getSarpropStepErrorThresholdFactor", getSarpropStepErrorThresholdFactor);
	Nan::SetPrototypeMethod(tpl, "getSarpropStepErrorShift", getSarpropStepErrorShift);
	Nan::SetPrototypeMethod(tpl, "getSarpropTemperature", getSarpropTemperature);
	Nan::SetPrototypeMethod(tpl, "getLearningMomentum", getLearningMomentum);
	Nan::SetPrototypeMethod(tpl, "setSarpropWeightDecayShift", setSarpropWeightDecayShift);
	Nan::SetPrototypeMethod(tpl, "setSarpropStepErrorThresholdFactor", setSarpropStepErrorThresholdFactor);
	Nan::SetPrototypeMethod(tpl, "setSarpropStepErrorShift", setSarpropStepErrorShift);
	Nan::SetPrototypeMethod(tpl, "setSarpropTemperature", setSarpropTemperature);
	Nan::SetPrototypeMethod(tpl, "setLearningMomentum", setLearningMomentum);

	Nan::SetPrototypeMethod(tpl, "getActivationSteepness", getActivationSteepness);
	Nan::SetPrototypeMethod(tpl, "setActivationSteepness", setActivationSteepness);
	Nan::SetPrototypeMethod(tpl, "setActivationSteepnessLayer", setActivationSteepnessLayer);
	Nan::SetPrototypeMethod(tpl, "setActivationSteepnessHidden", setActivationSteepnessHidden);
	Nan::SetPrototypeMethod(tpl, "setActivationSteepnessOutput", setActivationSteepnessOutput);

	Nan::SetPrototypeMethod(tpl, "setWeightArray", setWeightArray);
	Nan::SetPrototypeMethod(tpl, "setWeight", setWeight);

	Nan::SetPrototypeMethod(tpl, "getUserDataString", getUserDataString);
	Nan::SetPrototypeMethod(tpl, "setUserDataString", setUserDataString);

	// Create the loadFile function
	v8::Local<v8::FunctionTemplate> loadFileTpl = Nan::New<v8::FunctionTemplate>(loadFile);
	v8::Local<v8::Function> loadFileFunction = Nan::GetFunction(loadFileTpl).ToLocalChecked();

	v8::Local<v8::FunctionTemplate> disableSeedRandTpl = Nan::New<v8::FunctionTemplate>(disableSeedRand);
	v8::Local<v8::Function> disableSeedRandFunction = Nan::GetFunction(disableSeedRandTpl).ToLocalChecked();
	v8::Local<v8::FunctionTemplate> enableSeedRandTpl = Nan::New<v8::FunctionTemplate>(enableSeedRand);
	v8::Local<v8::Function> enableSeedRandFunction = Nan::GetFunction(enableSeedRandTpl).ToLocalChecked();

	// Assign a property called 'FANNY' to module.exports, pointing to our constructor
	v8::Local<v8::Function> ctorFunction = Nan::GetFunction(tpl).ToLocalChecked();
	Nan::Set(ctorFunction, Nan::New("loadFile").ToLocalChecked(), loadFileFunction);
	Nan::Set(ctorFunction, Nan::New("disableSeedRand").ToLocalChecked(), disableSeedRandFunction);
	Nan::Set(ctorFunction, Nan::New("enableSeedRand").ToLocalChecked(), enableSeedRandFunction);
	FANNY::constructorFunction.Reset(ctorFunction);
	Nan::Set(target, Nan::New("FANNY").ToLocalChecked(), ctorFunction);
}


Nan::Persistent<v8::FunctionTemplate> FANNY::constructorFunctionTpl;
Nan::Persistent<v8::Function> FANNY::constructorFunction;

FANNY::FANNY(FANN::neural_net *_fann) : fann(_fann), currentTrainWorker(NULL) {}

FANNY::~FANNY() {
	delete fann;
	constructorFunctionTpl.Empty();
	constructorFunction.Empty();
}

NAN_METHOD(FANNY::printConnections) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fanny->fann->print_connections();
}

NAN_METHOD(FANNY::printParameters) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fanny->fann->print_parameters();
}

NAN_METHOD(FANNY::randomizeWeights) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: min_weight and max_weight");

	if (!info[0]->IsNumber() || !info[1]->IsNumber()) {
		return Nan::ThrowError("min_weight and max_weight must be numbers");
	}
	fann_type min_weight = v8NumberToFannType(info[0]);
	fann_type max_weight = v8NumberToFannType(info[1]);

	fanny->fann->randomize_weights(min_weight, max_weight);
}

NAN_METHOD(FANNY::loadFile) {
	if (info.Length() != 2) return Nan::ThrowError("Requires filename and callback");
	std::string filename = *v8::String::Utf8Value(info[0]);
	Nan::Callback * callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new LoadFileWorker(callback, filename));
}

NAN_METHOD(FANNY::New) {
	// Ensure arguments
	if (info.Length() != 1) {
		return Nan::ThrowError("Requires single argument");
	}

	FANN::neural_net *fann;

	if (Nan::New(FANNY::constructorFunctionTpl)->HasInstance(info[0])) {
		// Copy constructor
		FANNY *other = Nan::ObjectWrap::Unwrap<FANNY>(info[0].As<v8::Object>());
		fann = new FANN::neural_net(*other->fann);
	} else if (info[0]->IsString()) {
		// Load-from-file constructor
		fann = new FANN::neural_net(std::string(*v8::String::Utf8Value(info[0])));
	} else if (info[0]->IsExternal()) {
		// Internal from-instance constructor
		fann = (FANN::neural_net *)info[0].As<v8::External>()->Value();
	} else if (info[0]->IsObject()) {
		// Options constructor

		// Get the options argument
		v8::Local<v8::Object> optionsObj(info[0].As<v8::Object>());

		// Variables for individual options
		std::string optType;
		std::vector<unsigned int> optLayers;
		float optConnectionRate = 0.5;

		// Get the type option
		Nan::MaybeLocal<v8::Value> maybeType = Nan::Get(optionsObj, Nan::New("type").ToLocalChecked());
		if (!maybeType.IsEmpty()) {
			v8::Local<v8::Value> localType = maybeType.ToLocalChecked();
			if (localType->IsString()) {
				optType = std::string(*(v8::String::Utf8Value(localType)));
			}
		}

		// Get the layers option
		Nan::MaybeLocal<v8::Value> maybeLayers = Nan::Get(optionsObj, Nan::New("layers").ToLocalChecked());
		if (!maybeLayers.IsEmpty()) {
			v8::Local<v8::Value> localLayers = maybeLayers.ToLocalChecked();
			if (localLayers->IsArray()) {
				v8::Local<v8::Array> arrayLayers = localLayers.As<v8::Array>();
				uint32_t length = arrayLayers->Length();
				for (uint32_t idx = 0; idx < length; ++idx) {
					Nan::MaybeLocal<v8::Value> maybeIdxValue = Nan::Get(arrayLayers, idx);
					if (!maybeIdxValue.IsEmpty()) {
						v8::Local<v8::Value> localIdxValue = maybeIdxValue.ToLocalChecked();
						if (localIdxValue->IsNumber()) {
							unsigned int idxValue = localIdxValue->Uint32Value();
							optLayers.push_back(idxValue);
						}
					}
				}
			}
		}
		if (optLayers.size() < 2) {
			return Nan::ThrowError("layers option is required with at least 2 layers");
		}

		// Get the connectionRate option
		Nan::MaybeLocal<v8::Value> maybeConnectionRate = Nan::Get(optionsObj, Nan::New("connectionRate").ToLocalChecked());
		if (!maybeConnectionRate.IsEmpty()) {
			v8::Local<v8::Value> localConnectionRate = maybeConnectionRate.ToLocalChecked();
			if (localConnectionRate->IsNumber()) {
				optConnectionRate = localConnectionRate->NumberValue();
			}
		}

		// Construct the neural_net underlying class
		if (!optType.compare("standard") || optType.empty()) {
			fann = new FANN::neural_net(FANN::network_type_enum::LAYER, (unsigned int)optLayers.size(), (const unsigned int *)&optLayers[0]);
		} else if(optType.compare("sparse")) {
			fann = new FANN::neural_net(optConnectionRate, optLayers.size(), &optLayers[0]);
		} else if (optType.compare("shortcut")) {
			fann = new FANN::neural_net(FANN::network_type_enum::SHORTCUT, optLayers.size(), &optLayers[0]);
		} else {
			return Nan::ThrowError("Invalid type option");
		}
	} else {
		return Nan::ThrowTypeError("Invalid argument type");
	}

	FANNY *obj = new FANNY(fann);
	obj->Wrap(info.This());
	info.GetReturnValue().Set(info.This());
}

bool FANNY::checkError() {
	unsigned int fannerr = fann->get_errno();
	if (fannerr) {
		std::string errstr = fann->get_errstr();
		std::string msg = std::string("FANN error ") + std::to_string(fannerr) + ": " + errstr;
		Nan::ThrowError(msg.c_str());
		fann->reset_errno();
		fann->reset_errstr();
		return true;
	} else {
		return false;
	}
}

NAN_METHOD(FANNY::save) {
	if (info.Length() != 2) return Nan::ThrowError("Takes a filename and a callback");
	if (!info[0]->IsString() || !info[1]->IsFunction()) return Nan::ThrowTypeError("Wrong argument type");
	std::string filename(*v8::String::Utf8Value(info[0]));
	Nan::Callback *callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new SaveFileWorker(callback, info.Holder(), filename, false));
}

NAN_METHOD(FANNY::saveToFixed) {
	if (info.Length() != 2) return Nan::ThrowError("Takes a filename and a callback");
	if (!info[0]->IsString() || !info[1]->IsFunction()) return Nan::ThrowTypeError("Wrong argument type");
	std::string filename(*v8::String::Utf8Value(info[0]));
	Nan::Callback *callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new SaveFileWorker(callback, info.Holder(), filename, true));
}

int FANNY::fannInternalCallback(
	FANN::neural_net &fann,
	FANN::training_data &train,
	unsigned int max_epochs,
	unsigned int epochs_between_reports,
	float desired_error,
	unsigned int epochs,
	void *user_data
) {
	FANNY *fanny = (FANNY *)user_data;
	fanny->currentTrainingProgress.iteration = epochs;
	fanny->currentTrainingProgress.mse = fanny->fann->get_MSE();
	fanny->currentTrainingProgress.bitFail = fanny->fann->get_bit_fail();
	if (fanny->currentTrainWorker && fanny->currentTrainWorker->executionProgress) {
		fanny->currentTrainWorker->executionProgress->Signal();
	}
	if (fanny->cancelTrainingFlag) {
		return -1;
	} else {
		return 1;
	}
}

NAN_METHOD(FANNY::setCallback) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() == 0 || !info[0]->IsFunction()) {
		fanny->trainingCallbackFn.Reset();
		fanny->fann->set_callback(NULL, NULL);
	} else {
		fanny->trainingCallbackFn.Reset(info[0].As<v8::Function>());
		fanny->fann->set_callback(fannInternalCallback, fanny);
	}
	#else
	Nan::ThrowError("Not supported for fixed FANN");
	#endif
}

void FANNY::_doTrainOrTest(
	const Nan::FunctionCallbackInfo<v8::Value> &info,
	bool fromFile,
	bool isCascade,
	bool singleEpoch,
	bool isTest
) {
	#ifndef FANNY_FIXED
	bool hasConfigParams = !singleEpoch && !isTest;
	int numArgs = hasConfigParams ? 5 : 2;
	if (info.Length() != numArgs) return Nan::ThrowError("Invalid arguments");
	std::string filename;
	Nan::MaybeLocal<v8::Object> maybeTrainingData;
	if (fromFile) {
		if (!info[0]->IsString()) return Nan::ThrowTypeError("First argument must be a string");
		filename = std::string(*v8::String::Utf8Value(info[0]));
	} else {
		if (!info[0]->IsObject()) return Nan::ThrowTypeError("First argument must be TrainingData");
		if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) return Nan::ThrowTypeError("First argument must be TrainingData");
		v8::Local<v8::Object> trainingDataHolder = info[0].As<v8::Object>();
		maybeTrainingData = Nan::MaybeLocal<v8::Object>(trainingDataHolder);
	}
	unsigned int maxIterations = 0;
	unsigned int iterationsBetweenReports = 0;
	float desiredError = 0;
	if (hasConfigParams) {
		if (!info[1]->IsNumber() || !info[2]->IsNumber() || !info[3]->IsNumber()) {
			return Nan::ThrowTypeError("Arguments must be numbers");
		}
		maxIterations = info[1]->Uint32Value();
		iterationsBetweenReports = info[2]->Uint32Value();
		desiredError = (float)info[3]->NumberValue();
	}
	if (!info[numArgs - 1]->IsFunction()) return Nan::ThrowTypeError("Last argument must be callback");
	Nan::Callback *callback = new Nan::Callback(info[numArgs - 1].As<v8::Function>());
	Nan::AsyncQueueWorker(new TrainWorker(
		callback,
		info.Holder(),
		maybeTrainingData,
		fromFile,
		filename,
		isCascade,
		maxIterations,
		iterationsBetweenReports,
		desiredError,
		singleEpoch,
		isTest
	));
	#else
	Nan::ThrowError("Not supported for fixed FANN");
	#endif
}

NAN_METHOD(FANNY::trainEpoch) {
	_doTrainOrTest(info, false, false, true, false);
}

NAN_METHOD(FANNY::trainOnData) {
	_doTrainOrTest(info, false, false, false, false);
}

NAN_METHOD(FANNY::trainOnFile) {
	_doTrainOrTest(info, true, false, false, false);
}

NAN_METHOD(FANNY::cascadetrainOnData) {
	_doTrainOrTest(info, false, true, false, false);
}

NAN_METHOD(FANNY::cascadetrainOnFile) {
	_doTrainOrTest(info, true, true, false, false);
}

NAN_METHOD(FANNY::testData) {
	_doTrainOrTest(info, false, false, true, true);
}

NAN_METHOD(FANNY::run) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Takes one argument");
	if (!info[0]->IsArray()) return Nan::ThrowError("Must be array");
	std::vector<fann_type> inputs = v8ArrayToFannData(info[0]);
	if (inputs.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	fann_type *outputs = fanny->fann->run(&inputs[0]);
	if (fanny->checkError()) return;
	v8::Local<v8::Value> outputArray = fannDataToV8Array(outputs, fanny->fann->get_num_output());
	info.GetReturnValue().Set(outputArray);
}

NAN_METHOD(FANNY::runAsync) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Takes two arguments");
	if (!info[0]->IsArray()) return Nan::ThrowError("First argument must be array");
	if (!info[1]->IsFunction()) return Nan::ThrowError("Second argument must be callback");
	std::vector<fann_type> inputs = v8ArrayToFannData(info[0]);
	if (inputs.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	Nan::Callback * callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new RunWorker(callback, inputs, info.Holder()));
}

NAN_METHOD(FANNY::getTrainingAlgorithm) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::training_algorithm_enum value = fanny->fann->get_training_algorithm();

	info.GetReturnValue().Set(trainingAlgorithmEnumToV8String(value));
}

NAN_METHOD(FANNY::setTrainingAlgorithm) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: training_algorithm");
	if (!info[0]->IsString()) return Nan::ThrowError("training_algorithm not a string");

	FANN::training_algorithm_enum value;
	if(v8StringToTrainingAlgorithmEnum(info[0], value)) fanny->fann->set_training_algorithm(value);
}

NAN_METHOD(FANNY::getTrainErrorFunction) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::error_function_enum value = fanny->fann->get_train_error_function();

	info.GetReturnValue().Set(errorFunctionEnumToV8String(value));
}

NAN_METHOD(FANNY::setTrainErrorFunction) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: train_error_function");
	if (!info[0]->IsString()) return Nan::ThrowError("train_error_function not a string");

	FANN::error_function_enum value;
	if(v8StringToErrorFunctionEnum(info[0], value)) fanny->fann->set_train_error_function(value);
}

NAN_METHOD(FANNY::getTrainStopFunction) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::stop_function_enum value = fanny->fann->get_train_stop_function();

	info.GetReturnValue().Set(stopFunctionEnumToV8String(value));
}

NAN_METHOD(FANNY::setTrainStopFunction) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: train_stop_function");
	if (!info[0]->IsString()) return Nan::ThrowError("train_stop_function not a string");

	FANN::stop_function_enum value;
	if(v8StringToStopFunctionEnum(info[0], value)) fanny->fann->set_train_stop_function(value);
}

NAN_METHOD(FANNY::getNumInput) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_input();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getNumOutput) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_output();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getTotalNeurons) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_total_neurons();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getTotalConnections) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_total_connections();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getConnectionArray) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int size = fanny->fann->get_total_connections();
	std::vector<FANN::connection> connections(size);
	fanny->fann->get_connection_array(&connections[0]);

	info.GetReturnValue().Set(connectionArrayToV8Array(connections));
}

NAN_METHOD(FANNY::getNumLayers) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_layers();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getBitFail) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_bit_fail();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getBitFailLimit) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fann_type num = fanny->fann->get_bit_fail_limit();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setBitFailLimit) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: bit_fail_limit");
	if (!info[0]->IsNumber()) return Nan::ThrowError("bit_fail_limit not a number");

	fann_type value = v8NumberToFannType(info[0]);

	fanny->fann->set_bit_fail_limit(value);
}

NAN_METHOD(FANNY::getMSE) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_MSE();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::resetMSE) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fanny->fann->reset_MSE();
}

// by default 0.7
NAN_METHOD(FANNY::getLearningRate) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_learning_rate();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setLearningRate) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: learning_rate");
	if (!info[0]->IsNumber()) return Nan::ThrowError("learning_rate not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_learning_rate(value);
}

NAN_METHOD(FANNY::getActivationFunction) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: layer and neuron");
	if (!info[0]->IsNumber()) return Nan::ThrowError("layer must be a number");
	if (!info[1]->IsNumber()) return Nan::ThrowError("neuron must be a number");

	unsigned int layer = info[0]->Uint32Value();
	unsigned int neuron = info[1]->Uint32Value();
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum activationFunction = fanny->fann->get_activation_function(layer, neuron);
	info.GetReturnValue().Set(activationFunctionEnumToV8String(activationFunction));
}

NAN_METHOD(FANNY::setActivationFunction) {
	if (info.Length() != 3) return Nan::ThrowError("Must have 3 arguments: activation_function, layer, and neuron");
	if (!info[0]->IsString()) return Nan::ThrowError("activation_function must be a string");
	if (!info[1]->IsNumber()) return Nan::ThrowError("layer must be a number");
	if (!info[2]->IsNumber()) return Nan::ThrowError("neuron must be a number");

	unsigned int layer = info[1]->Uint32Value();
	unsigned int neuron = info[2]->Uint32Value();

	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum activationFunction;
	if(v8StringToActivationFunctionEnum(info[0], activationFunction)) fanny->fann->set_activation_function(activationFunction, layer, neuron);
}

NAN_METHOD(FANNY::setActivationFunctionLayer) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: activation_function, layer");
	if (!info[0]->IsString()) return Nan::ThrowError("activation_function must be a string");
	if (!info[1]->IsNumber()) return Nan::ThrowError("layer must be a number");

	unsigned int layer = info[1]->Uint32Value();

	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum activationFunction;
	if(v8StringToActivationFunctionEnum(info[0], activationFunction)) fanny->fann->set_activation_function_layer(activationFunction, layer);
}

NAN_METHOD(FANNY::setActivationFunctionHidden) {
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: activation_function,");
	if (!info[0]->IsString()) return Nan::ThrowError("activation_function must be a string");

	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum activationFunction;
	if(v8StringToActivationFunctionEnum(info[0], activationFunction)) fanny->fann->set_activation_function_hidden(activationFunction);
}

NAN_METHOD(FANNY::setActivationFunctionOutput) {
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: activation_function,");
	if (!info[0]->IsString()) return Nan::ThrowError("activation_function must be a string");

	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum activationFunction;
	if(v8StringToActivationFunctionEnum(info[0], activationFunction)) fanny->fann->set_activation_function_output(activationFunction);
}

// by default -0.0001
NAN_METHOD(FANNY::getQuickpropDecay) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_quickprop_decay();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setQuickpropDecay) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: quickprop_decay");
	if (!info[0]->IsNumber()) return Nan::ThrowError("quickprop_decay not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_quickprop_decay(value);
}

// by default 1.75
NAN_METHOD(FANNY::getQuickpropMu) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_quickprop_mu();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setQuickpropMu) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: quickprop_mu");
	if (!info[0]->IsNumber()) return Nan::ThrowError("quickprop_mu not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_quickprop_mu(value);
}

// by default 1.2
NAN_METHOD(FANNY::getRpropIncreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_increase_factor();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setRpropIncreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: rprop_increase_factor");
	if (!info[0]->IsNumber()) return Nan::ThrowError("rprop_increase_factor not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_rprop_increase_factor(value);
}

// by default 0.5
NAN_METHOD(FANNY::getRpropDecreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_decrease_factor();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setRpropDecreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: rprop_decrease_factor");
	if (!info[0]->IsNumber()) return Nan::ThrowError("rprop_decrease_factor not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_rprop_decrease_factor(value);
}

// by default 0.1
NAN_METHOD(FANNY::getRpropDeltaZero) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_zero();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setRpropDeltaZero) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: rprop_delta_zero");
	if (!info[0]->IsNumber()) return Nan::ThrowError("rprop_delta_zero not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_rprop_delta_zero(value);
}

// by default 0.0
NAN_METHOD(FANNY::getRpropDeltaMin) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_min();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setRpropDeltaMin) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: rprop_delta_min");
	if (!info[0]->IsNumber()) return Nan::ThrowError("rprop_delta_min not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_rprop_delta_min(value);
}

// by default 50.0
NAN_METHOD(FANNY::getRpropDeltaMax) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_max();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setRpropDeltaMax) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: rprop_delta_max");
	if (!info[0]->IsNumber()) return Nan::ThrowError("rprop_delta_max not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_rprop_delta_max(value);
}

// The default delta max is -6.644
NAN_METHOD(FANNY::getSarpropWeightDecayShift) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_sarprop_weight_decay_shift();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setSarpropWeightDecayShift) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: sarprop_weight_decay_shift");
	if (!info[0]->IsNumber()) return Nan::ThrowError("sarprop_weight_decay_shift not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_sarprop_weight_decay_shift(value);
}

// The default delta max is 0.1
NAN_METHOD(FANNY::getSarpropStepErrorThresholdFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_sarprop_step_error_threshold_factor();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setSarpropStepErrorThresholdFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: sarprop_step_error_threshold_factor");
	if (!info[0]->IsNumber()) return Nan::ThrowError("sarprop_step_error_threshold_factor not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_sarprop_step_error_threshold_factor(value);
}

// The default delta max is 1.385
NAN_METHOD(FANNY::getSarpropStepErrorShift) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_sarprop_step_error_shift();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setSarpropStepErrorShift) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: sarprop_step_error_shift");
	if (!info[0]->IsNumber()) return Nan::ThrowError("sarprop_step_error_shift not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_sarprop_step_error_shift(value);
}

// The default delta max is 0.015
NAN_METHOD(FANNY::getSarpropTemperature) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_sarprop_temperature();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setSarpropTemperature) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: sarprop_temperature");
	if (!info[0]->IsNumber()) return Nan::ThrowError("sarprop_temperature not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_sarprop_temperature(value);
}

// The default is 0
NAN_METHOD(FANNY::getLearningMomentum) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_learning_momentum();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::setLearningMomentum) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: learning_momentum");
	if (!info[0]->IsNumber()) return Nan::ThrowError("learning_momentum not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_learning_momentum(value);
}

NAN_METHOD(FANNY::initWeights) {
	if (info.Length() != 1) return Nan::ThrowError("Takes an argument");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}
	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	return fanny->fann->init_weights(*fannyTrainingData->trainingData);
}

NAN_METHOD(FANNY::getLayerArray) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	std::vector<unsigned int> retArrayVector(fanny->fann->get_num_layers());
	fanny->fann->get_layer_array(&retArrayVector[0]);
	uint32_t size = retArrayVector.size();
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		v8::Local<v8::Value> value = Nan::New<v8::Number>(retArrayVector[idx]);
		Nan::Set(v8Array, idx, value);
	}
	info.GetReturnValue().Set(v8Array);
}

NAN_METHOD(FANNY::getBiasArray) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	std::vector<unsigned int> retArrayVector(fanny->fann->get_num_layers());
	fanny->fann->get_bias_array(&retArrayVector[0]);
	uint32_t size = retArrayVector.size();
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		v8::Local<v8::Value> value = Nan::New<v8::Number>(retArrayVector[idx]);
		Nan::Set(v8Array, idx, value);
	}
	info.GetReturnValue().Set(v8Array);
}

NAN_METHOD(FANNY::train) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: input, desired_output");
	if (!info[0]->IsArray() || !info[1]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> input = v8ArrayToFannData(info[0]);
	std::vector<fann_type> desired_output = v8ArrayToFannData(info[1]);

	if (input.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	if (desired_output.size() != fanny->fann->get_num_output()) return Nan::ThrowError("Wrong number of desired ouputs");

	fanny->fann->train(&input[0], &desired_output[0]);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::test) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: input, desired_output");
	if (!info[0]->IsArray() || !info[1]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> input = v8ArrayToFannData(info[0]);
	std::vector<fann_type> desired_output = v8ArrayToFannData(info[1]);

	if (input.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	if (desired_output.size() != fanny->fann->get_num_output()) return Nan::ThrowError("Wrong number of desired ouputs");

	fann_type *outputs = 	fanny->fann->test(&input[0], &desired_output[0]);
	if (fanny->checkError()) return;
	v8::Local<v8::Value> outputArray = fannDataToV8Array(outputs, fanny->fann->get_num_output());
	info.GetReturnValue().Set(outputArray);
}

NAN_METHOD(FANNY::scaleTrain) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an argument: tainingData");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}
	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());
	fanny->fann->scale_train(*fannyTrainingData->trainingData);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::descaleTrain) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an argument: tainingData");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}
	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());
	fanny->fann->descale_train(*fannyTrainingData->trainingData);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::clearScalingParams) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	bool result = fanny->fann->clear_scaling_params();

	info.GetReturnValue().Set(result);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setInputScalingParams) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 3) return Nan::ThrowError("Must have 3 arguments: tainingData, new_input_min, and new_input_max");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}

	if (!info[1]->IsNumber() || !info[2]->IsNumber()) {
		return Nan::ThrowError("new_input_min and new_input_max must be numbers");
	}

	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());

	float new_input_min = info[1]->NumberValue();
	float new_input_max = info[2]->NumberValue();


	fanny->fann->set_input_scaling_params(*fannyTrainingData->trainingData, new_input_min, new_input_max);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setOutputScalingParams) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 3) return Nan::ThrowError("Must have 3 arguments: tainingData new_output_min, and new_output_max");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}

	if (!info[1]->IsNumber() || !info[2]->IsNumber()) {
		return Nan::ThrowError("new_output_min and new_output_max must be numbers");
	}

	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());

	float new_output_min = info[1]->NumberValue();
	float new_output_max = info[2]->NumberValue();

	fanny->fann->set_output_scaling_params(*fannyTrainingData->trainingData, new_output_min, new_output_max);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setScalingParams) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 5) return Nan::ThrowError("Must have 5 arguments: tainingData, new_input_min, new_input_max, new_output_min, and new_output_max");
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Argument must be an instance of TrainingData");
	}

	if (!info[1]->IsNumber() || !info[2]->IsNumber() || !info[3]->IsNumber() || !info[4]->IsNumber()) {
		return Nan::ThrowError("new_input_min, new_input_max, new_output_min, and new_output_max must be numbers");
	}

	TrainingData *fannyTrainingData = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());

	float new_input_min = info[1]->NumberValue();
	float new_input_max = info[2]->NumberValue();
	float new_output_min = info[3]->NumberValue();
	float new_output_max = info[4]->NumberValue();

	fanny->fann->set_scaling_params(*fannyTrainingData->trainingData, new_input_min, new_input_max, new_output_min, new_output_max);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::scaleInput) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: input_vector");
	if (!info[0]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> input = v8ArrayToFannData(info[0]);

	if (input.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");

	fanny->fann->scale_input(&input[0]);

	info.GetReturnValue().Set(fannDataToV8Array(&input[0], fanny->fann->get_num_input()));

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::scaleOutput) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: output_vector");
	if (!info[0]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> output = v8ArrayToFannData(info[0]);

	if (output.size() != fanny->fann->get_num_output()) return Nan::ThrowError("Wrong number of outputs");

	fanny->fann->scale_output(&output[0]);

	info.GetReturnValue().Set(fannDataToV8Array(&output[0], fanny->fann->get_num_output()));

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::descaleInput) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: input_vector");
	if (!info[0]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> input = v8ArrayToFannData(info[0]);

	if (input.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");

	fanny->fann->descale_input(&input[0]);

	info.GetReturnValue().Set(fannDataToV8Array(&input[0], fanny->fann->get_num_input()));

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::descaleOutput) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: output_vector");
	if (!info[0]->IsArray()) return Nan::ThrowError("Argument not an array");

	std::vector<fann_type> output = v8ArrayToFannData(info[0]);

	if (output.size() != fanny->fann->get_num_output()) return Nan::ThrowError("Wrong number of outputs");

	fanny->fann->descale_output(&output[0]);

	info.GetReturnValue().Set(fannDataToV8Array(&output[0], fanny->fann->get_num_output()));

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeActivationFunctions) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	FANN::activation_function_enum* activationFunctions = fanny->fann->get_cascade_activation_functions();
	uint32_t size = fanny->fann->get_cascade_activation_functions_count();
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		const char *str = NULL;
		FANN::activation_function_enum value = activationFunctions[idx];
		switch(value) {
			case FANN::LINEAR: str = "LINEAR"; break;
			case FANN::THRESHOLD: str = "THRESHOLD"; break;
			case FANN::THRESHOLD_SYMMETRIC: str = "THRESHOLD_SYMMETRIC"; break;
			case FANN::SIGMOID: str = "SIGMOID"; break;
			case FANN::SIGMOID_STEPWISE: str = "SIGMOID_STEPWISE"; break;
			case FANN::SIGMOID_SYMMETRIC: str = "SIGMOID_SYMMETRIC"; break;
			case FANN::SIGMOID_SYMMETRIC_STEPWISE: str = "SIGMOID_SYMMETRIC_STEPWISE"; break;
			case FANN::GAUSSIAN: str = "GAUSSIAN"; break;
			case FANN::GAUSSIAN_SYMMETRIC: str = "GAUSSIAN_SYMMETRIC"; break;
			case FANN::GAUSSIAN_STEPWISE: str = "GAUSSIAN_STEPWISE"; break;
			case FANN::ELLIOT: str = "ELLIOT"; break;
			case FANN::ELLIOT_SYMMETRIC: str = "ELLIOT_SYMMETRIC"; break;
			case FANN::LINEAR_PIECE: str = "LINEAR_PIECE"; break;
			case FANN::LINEAR_PIECE_SYMMETRIC: str = "LINEAR_PIECE_SYMMETRIC"; break;
			case FANN::SIN_SYMMETRIC: str = "SIN_SYMMETRIC"; break;
			case FANN::COS_SYMMETRIC: str = "COS_SYMMETRIC"; break;
			case FANN::COS: str = "COS"; break;
			case FANN::SIN: str = "SIN"; break;
		}
		v8::Local<v8::Value> ret;
		if (str) {
			ret = Nan::New(str).ToLocalChecked();
			Nan::Set(v8Array, idx, ret);
		}
	}
	info.GetReturnValue().Set(v8Array);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeActivationFunctions) {
	#ifndef FANNY_FIXED
	if (!info[0]->IsArray()) return Nan::ThrowError("1st argument must be an array");
	if (!info[1]->IsNumber()) return Nan::ThrowError("2nd argument must be a number");
	v8::Local<v8::Array> inputs = info[0].As<v8::Array>();
	unsigned int size = info[1]->Uint32Value();
	if (inputs->Length() != size) return Nan::ThrowError("The length of the array must be the second argument");
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	std::vector<FANN::activation_function_enum> activationFunctions;
	for (uint32_t idx = 0; idx < size; ++idx) {
		Nan::MaybeLocal<v8::Value> maybeIdxValue = Nan::Get(inputs, idx);
		if (!maybeIdxValue.IsEmpty()) {
			v8::Local<v8::Value> value = maybeIdxValue.ToLocalChecked();
			if (value->IsString()) {
				FANN::activation_function_enum ret;
				std::string str = std::string(*(v8::String::Utf8Value(value)));
				if (str.compare("LINEAR") == 0) ret = FANN::LINEAR;
				else if (str.compare("THRESHOLD") == 0) ret = FANN::THRESHOLD;
				else if (str.compare("THRESHOLD_SYMMETRIC") == 0) ret = FANN::THRESHOLD_SYMMETRIC;
				else if (str.compare("SIGMOID") == 0) ret = FANN::SIGMOID;
				else if (str.compare("SIGMOID_STEPWISE") == 0) ret = FANN::SIGMOID_STEPWISE;
				else if (str.compare("SIGMOID_SYMMETRIC") == 0) ret = FANN::SIGMOID_SYMMETRIC;
				else if (str.compare("SIGMOID_SYMMETRIC_STEPWISE") == 0) ret = FANN::SIGMOID_SYMMETRIC_STEPWISE;
				else if (str.compare("GAUSSIAN") == 0) ret = FANN::GAUSSIAN;
				else if (str.compare("GAUSSIAN_SYMMETRIC") == 0) ret = FANN::GAUSSIAN_SYMMETRIC;
				else if (str.compare("GAUSSIAN_STEPWISE") == 0) ret = FANN::GAUSSIAN_STEPWISE;
				else if (str.compare("ELLIOT") == 0) ret = FANN::ELLIOT;
				else if (str.compare("ELLIOT_SYMMETRIC") == 0) ret = FANN::ELLIOT_SYMMETRIC;
				else if (str.compare("LINEAR_PIECE") == 0) ret = FANN::LINEAR_PIECE;
				else if (str.compare("LINEAR_PIECE_SYMMETRIC") == 0) ret = FANN::LINEAR_PIECE_SYMMETRIC;
				else if (str.compare("SIN_SYMMETRIC") == 0) ret = FANN::SIN_SYMMETRIC;
				else if (str.compare("COS_SYMMETRIC") == 0) ret = FANN::COS_SYMMETRIC;
				else if (str.compare("COS") == 0) ret = FANN::COS;
				else if (str.compare("SIN") == 0) ret = FANN::SIN;
				else continue;
				activationFunctions.push_back(ret);
			}
		}
	}

	if (activationFunctions.size() != size) {
		return Nan::ThrowError("Some activation functions where not found");
	} else {
		fanny->fann->set_cascade_activation_functions(&activationFunctions[0], size);
	}

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeActivationSteepnesses) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	fann_type *cascadeActivationSteepnesses = fanny->fann->get_cascade_activation_steepnesses();

	info.GetReturnValue().Set(fannDataToV8Array(cascadeActivationSteepnesses, fanny->fann->get_cascade_activation_steepnesses_count()));

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeActivationSteepnesses) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: 	cascade_activation_steepnesses and cascade_activation_steepnesses_count");
	if (!info[0]->IsArray()) return Nan::ThrowError("cascade_activation_steepnesse not an array");
	if (!info[1]->IsNumber()) return Nan::ThrowError("cascade_activation_steepnesses_count not a number");

	std::vector<fann_type> cascadeActivationSteepnesses = v8ArrayToFannData(info[0]);
	unsigned int cascadeActivationSteepnessesCount = info[1]->Uint32Value();

	if (cascadeActivationSteepnesses.size() != cascadeActivationSteepnessesCount) return Nan::ThrowError("Wrong number of steepnesses or count");

	fanny->fann->set_cascade_activation_steepnesses(&cascadeActivationSteepnesses[0], cascadeActivationSteepnessesCount);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeOutputChangeFraction) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	float num = fanny->fann->get_cascade_output_change_fraction();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeOutputChangeFraction) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_output_change_fraction");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_output_change_fraction not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_cascade_output_change_fraction(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeOutputStagnationEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	unsigned int num = fanny->fann->get_cascade_output_stagnation_epochs();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeOutputStagnationEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_output_stagnation_epochs");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_output_stagnation_epochs not a number");

	unsigned int value = info[0]->Uint32Value();

	fanny->fann->set_cascade_output_stagnation_epochs(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeCandidateChangeFraction) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	float num = fanny->fann->get_cascade_candidate_change_fraction();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeCandidateChangeFraction) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_candidate_change_fraction");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_candidate_change_fraction not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_cascade_candidate_change_fraction(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeCandidateStagnationEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	unsigned int num = fanny->fann->get_cascade_candidate_stagnation_epochs();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeCandidateStagnationEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_candidate_stagnation_epochs");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_candidate_stagnation_epochs not a number");

	unsigned int value = info[0]->Uint32Value();

	fanny->fann->set_cascade_candidate_stagnation_epochs(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeWeightMultiplier) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fann_type num = fanny->fann->get_cascade_weight_multiplier();
	info.GetReturnValue().Set(num);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeWeightMultiplier) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_weight_multiplier");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_weight_multiplier not a number");

	fann_type value = v8NumberToFannType(info[0]);

	fanny->fann->set_cascade_weight_multiplier(value);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeCandidateLimit) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	fann_type num = fanny->fann->get_cascade_candidate_limit();
	info.GetReturnValue().Set(num);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeCandidateLimit) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_candidate_limit");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_candidate_limit not a number");

	fann_type value = v8NumberToFannType(info[0]);

	fanny->fann->set_cascade_candidate_limit(value);
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeMaxOutEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	unsigned int num = fanny->fann->get_cascade_max_out_epochs();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeMaxOutEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_max_out_epochs");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_max_out_epochs not a number");

	unsigned int value = info[0]->Uint32Value();

	fanny->fann->set_cascade_max_out_epochs(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeMaxCandEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	unsigned int num = fanny->fann->get_cascade_max_cand_epochs();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeMaxCandEpochs) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_max_cand_epochs");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_max_cand_epochs not a number");

	unsigned int value = info[0]->Uint32Value();

	fanny->fann->set_cascade_max_cand_epochs(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getCascadeNumCandidateGroups) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());

	unsigned int num = fanny->fann->get_cascade_num_candidate_groups();

	info.GetReturnValue().Set(num);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::setCascadeNumCandidateGroups) {
	#ifndef FANNY_FIXED
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have an arguments: cascade_num_candidate_groups");
	if (!info[0]->IsNumber()) return Nan::ThrowError("cascade_num_candidate_groups not a number");

	float value = info[0]->NumberValue();

	fanny->fann->set_cascade_num_candidate_groups(value);

	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(FANNY::getActivationSteepness) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have an arguments: layer and neuron");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) return Nan::ThrowError("layer and neuron should be numbers");
	unsigned int layer = info[0]->Uint32Value();
	unsigned int neuron = info[1]->Uint32Value();
	fann_type activationSteepness = fanny->fann->get_activation_steepness(layer, neuron);
	info.GetReturnValue().Set(activationSteepness);
}

NAN_METHOD(FANNY::setActivationSteepness) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 3) return Nan::ThrowError("Must have 3 arguments");
	if (!info[0]->IsNumber() || !info[1]->IsNumber() || !info[2]->IsNumber()) {
		return Nan::ThrowError("All arguments must be numbers");
	}

	fann_type steepness = v8NumberToFannType(info[0]);
	unsigned int layer = info[1]->Uint32Value();
	unsigned int neuron = info[2]->Uint32Value();
	fanny->fann->set_activation_steepness(steepness, layer, neuron);
}

NAN_METHOD(FANNY::setActivationSteepnessLayer) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) {
		return Nan::ThrowError("All arguments must be numbers");
	}

	fann_type steepness = v8NumberToFannType(info[0]);
	unsigned int layer = info[1]->Uint32Value();
	fanny->fann->set_activation_steepness_layer(steepness, layer);
}

NAN_METHOD(FANNY::setActivationSteepnessHidden) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have 1 arguments");
	if (!info[0]->IsNumber()) {
		return Nan::ThrowError("steepness must be numbers");
	}
	fann_type steepness = v8NumberToFannType(info[0]);
	fanny->fann->set_activation_steepness_hidden(steepness);
}

NAN_METHOD(FANNY::setActivationSteepnessOutput) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Must have 1 arguments");
	if (!info[0]->IsNumber()) {
		return Nan::ThrowError("steepness must be numbers");
	}
	fann_type steepness = v8NumberToFannType(info[0]);
	fanny->fann->set_activation_steepness_output(steepness);
}

NAN_METHOD(FANNY::setWeightArray) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Must have two arguments");
	if (!info[0]->IsArray()) return Nan::ThrowError("Connections must be an array");
	if (!info[1]->IsNumber()) return Nan::ThrowError("size must be a number");

	std::vector<FANN::connection> connections = v8ArrayToConnection(info[0]);
	unsigned int num = info[1]->Uint32Value();
	fanny->fann->set_weight_array(&connections[0], num);
}

NAN_METHOD(FANNY::setWeight) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 3) return Nan::ThrowError("Must have 3 arguments");
	if (!info[0]->IsNumber() || !info[1]->IsNumber() || !info[2]->IsNumber()) {
		return Nan::ThrowError("All arguments must be numbers");
	}
	unsigned int fromNeuron = info[0]->Uint32Value();
	unsigned int toNeuron = info[1]->Uint32Value();
	fann_type weight = v8NumberToFannType(info[2]);
	fanny->fann->set_weight(fromNeuron, toNeuron, weight);
}

NAN_METHOD(FANNY::getUserDataString) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	char *str = fanny->fann->get_user_data_string();
	if (str) {
		info.GetReturnValue().Set(Nan::New(str).ToLocalChecked());
	} else {
		info.GetReturnValue().Set(Nan::Null());
	}
}

NAN_METHOD(FANNY::setUserDataString) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1 || !info[0]->IsString()) return Nan::ThrowError("Argument must be string");
	v8::String::Utf8Value utf8String(info[0]);
	fanny->fann->set_user_data_string(*utf8String);
}

NAN_METHOD(FANNY::disableSeedRand) {
	fann_disable_seed_rand();
}

NAN_METHOD(FANNY::enableSeedRand) {
	fann_enable_seed_rand();
}

}

