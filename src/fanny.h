#ifndef FANNY_H
#define FANNY_H

#include <nan.h>
#include "fann-includes.h"

namespace fanny {

class TrainWorker;

class FANNY : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static void Init(v8::Local<v8::Object> target);

	// Encapsulated FANN neural_net instance
	FANN::neural_net *fann;

	// User-defined training callback function
	Nan::Persistent<v8::Function> trainingCallbackFn;

	// Constructor
	static Nan::Persistent<v8::FunctionTemplate> constructorFunctionTpl;
	static Nan::Persistent<v8::Function> constructorFunction;

	// Current training iteration data
	class TrainingProgress {
	public:
		unsigned int iteration;
		float mse;
		unsigned int bitFail;
	};
	TrainingProgress currentTrainingProgress;
	TrainWorker *currentTrainWorker;
	bool cancelTrainingFlag;

private:

	// Javascript Constructor.  Takes single "options" object parameter.  Options can include:
	// - type (string) - One of "standard", "sparse", "shortcut"
	// - layers (array of numbers)
	// - connectionRate (number) - For sparse networks
	static NAN_METHOD(New);

	static NAN_METHOD(loadFile);

	static NAN_METHOD(save);
	static NAN_METHOD(saveToFixed);

	// FANN "run" method.  Parameter is array of numbers.  Returns array of numbers.
	// Also takes a callback.
	static NAN_METHOD(runAsync);

	// Synchronous version of "run".
	static NAN_METHOD(run);
	static NAN_METHOD(getNumInput);
	static NAN_METHOD(getNumOutput);
	static NAN_METHOD(getTotalNeurons);
	static NAN_METHOD(getTotalConnections);
	static NAN_METHOD(getConnectionArray);
	static NAN_METHOD(getNumLayers);
	static NAN_METHOD(getBitFail);
	static NAN_METHOD(getBitFailLimit);
	static NAN_METHOD(setBitFailLimit);
	static NAN_METHOD(getMSE);
	static NAN_METHOD(resetMSE);
	static NAN_METHOD(getLearningRate);
	static NAN_METHOD(setLearningRate);
	static NAN_METHOD(getActivationFunction);
	static NAN_METHOD(setActivationFunction);
	static NAN_METHOD(setActivationFunctionLayer);
	static NAN_METHOD(setActivationFunctionHidden);
	static NAN_METHOD(setActivationFunctionOutput);
	static NAN_METHOD(getQuickpropDecay);
	static NAN_METHOD(getQuickpropMu);
	static NAN_METHOD(getRpropIncreaseFactor);
	static NAN_METHOD(getRpropDecreaseFactor);
	static NAN_METHOD(getRpropDeltaZero);
	static NAN_METHOD(getRpropDeltaMin);
	static NAN_METHOD(getRpropDeltaMax);
	static NAN_METHOD(initWeights);
	static NAN_METHOD(getLayerArray);
	static NAN_METHOD(getBiasArray);
	static NAN_METHOD(getCascadeActivationFunctions);
	static NAN_METHOD(setCascadeActivationFunctions);
	static NAN_METHOD(getCascadeActivationSteepnesses);
	static NAN_METHOD(setCascadeActivationSteepnesses);

	static NAN_METHOD(printConnections);
	static NAN_METHOD(printParameters);

	static NAN_METHOD(randomizeWeights);

	// TODO
	/*
	- train (rename to trainOne)
	- getActivationSteepness
	- setActivationSteepness
	- setActivationSteepnessLayer
	- setActivationSteepnessHidden
	- setActivationSteepnessOutput
	- setWeightArray
	- setWeight
	*/

	static NAN_METHOD(getCascadeOutputChangeFraction);
	static NAN_METHOD(getCascadeOutputStagnationEpochs);
	static NAN_METHOD(getCascadeCandidateChangeFraction);
	static NAN_METHOD(getCascadeCandidateStagnationEpochs);
	static NAN_METHOD(getCascadeWeightMultiplier);
	static NAN_METHOD(getCascadeCandidateLimit);
	static NAN_METHOD(getCascadeMaxOutEpochs);
	static NAN_METHOD(getCascadeMaxCandEpochs);
	static NAN_METHOD(getCascadeNumCandidateGroups);

	static NAN_METHOD(setCascadeOutputChangeFraction);
	static NAN_METHOD(setCascadeOutputStagnationEpochs);
	static NAN_METHOD(setCascadeCandidateChangeFraction);
	static NAN_METHOD(setCascadeCandidateStagnationEpochs);
	static NAN_METHOD(setCascadeWeightMultiplier);
	static NAN_METHOD(setCascadeCandidateLimit);
	static NAN_METHOD(setCascadeMaxOutEpochs);
	static NAN_METHOD(setCascadeMaxCandEpochs);
	static NAN_METHOD(setCascadeNumCandidateGroups);

	static NAN_METHOD(setQuickpropDecay);
	static NAN_METHOD(setQuickpropMu);
	static NAN_METHOD(setRpropIncreaseFactor);
	static NAN_METHOD(setRpropDecreaseFactor);
	static NAN_METHOD(setRpropDeltaZero);
	static NAN_METHOD(setRpropDeltaMin);
	static NAN_METHOD(setRpropDeltaMax);

	static NAN_METHOD(train);
	static NAN_METHOD(test);
	static NAN_METHOD(scaleTrain);
	static NAN_METHOD(descaleTrain);
	static NAN_METHOD(clearScalingParams);
	static NAN_METHOD(setInputScalingParams);
	static NAN_METHOD(setOutputScalingParams);
	static NAN_METHOD(setScalingParams);
	static NAN_METHOD(scaleInput);
	static NAN_METHOD(scaleOutput);
	static NAN_METHOD(descaleInput);
	static NAN_METHOD(descaleOutput);

	static NAN_METHOD(getTrainingAlgorithm);
	static NAN_METHOD(setTrainingAlgorithm);
	static NAN_METHOD(getTrainErrorFunction);
	static NAN_METHOD(setTrainErrorFunction);
	static NAN_METHOD(getTrainStopFunction);
	static NAN_METHOD(setTrainStopFunction);

	static NAN_METHOD(trainEpoch);
	static NAN_METHOD(trainOnData);
	static NAN_METHOD(trainOnFile);
	static NAN_METHOD(cascadetrainOnData);
	static NAN_METHOD(cascadetrainOnFile);
	static NAN_METHOD(testData);

	static NAN_METHOD(getSarpropWeightDecayShift);
	static NAN_METHOD(getSarpropStepErrorThresholdFactor);
	static NAN_METHOD(getSarpropStepErrorShift);
	static NAN_METHOD(getSarpropTemperature);
	static NAN_METHOD(getLearningMomentum);
	static NAN_METHOD(setSarpropWeightDecayShift);
	static NAN_METHOD(setSarpropStepErrorThresholdFactor);
	static NAN_METHOD(setSarpropStepErrorShift);
	static NAN_METHOD(setSarpropTemperature);
	static NAN_METHOD(setLearningMomentum);

	static NAN_METHOD(getActivationSteepness);

	static void _doTrainOrTest(const Nan::FunctionCallbackInfo<v8::Value> &info, bool fromFile, bool isCascade, bool singleEpoch, bool isTest);

	static NAN_METHOD(setCallback);
	static int fannInternalCallback(
		FANN::neural_net &fann,
		FANN::training_data &train,
		unsigned int max_epochs,
		unsigned int epochs_between_reports,
		float desired_error,
		unsigned int epochs,
		void *user_data
	);

	// constructor & destructor
	explicit FANNY(FANN::neural_net *fann);
	~FANNY();

	// Checks to see if the fann instance has a recorded error.
	// If so, this throws a JS error and returns true.
	// In this case, the calling function must return immediately.
	// It also resets the fann error.
	bool checkError();

};

}

#endif
