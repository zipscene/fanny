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
	static NAN_METHOD(getNumLayers);
	static NAN_METHOD(getBitFail);
	static NAN_METHOD(getMSE);
	static NAN_METHOD(getLearningRate);
	static NAN_METHOD(getQuickPropDecay);
	static NAN_METHOD(getQuickPropMu);
	static NAN_METHOD(getRpropIncreaseFactor);
	static NAN_METHOD(getRpropDecreaseFactor);
	static NAN_METHOD(getRpropDeltaZero);
	static NAN_METHOD(getRpropDeltaMin);
	static NAN_METHOD(getRpropDeltaMax);
	static NAN_METHOD(initWeights);
	static NAN_METHOD(getLayerArray);
	static NAN_METHOD(getBiasArray);
	static NAN_METHOD(train);
	static NAN_METHOD(test);

	static NAN_METHOD(trainEpoch);
	static NAN_METHOD(trainOnData);
	static NAN_METHOD(trainOnFile);
	static NAN_METHOD(cascadetrainOnData);
	static NAN_METHOD(cascadetrainOnFile);
	static NAN_METHOD(testData);

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
