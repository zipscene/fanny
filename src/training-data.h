#include <nan.h>
#include "fann-includes.h"

namespace fanny {

class TrainingData : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static void Init(v8::Local<v8::Object> target);

	// Encapsulated FANN training_data instance
	FANN::training_data *trainingData;

	// Reference to the javascript constructor FunctionTemplate
	static Nan::Persistent<v8::FunctionTemplate> constructorFunctionTpl;

private:

	// Javascript Constructor.  Takes no arguments.
	static NAN_METHOD(New);

	// constructor & destructor
	explicit TrainingData(FANN::training_data *training_data);
	~TrainingData();

	// Methods
	static NAN_METHOD(shuffle);
	static NAN_METHOD(merge);
	static NAN_METHOD(length);
	static NAN_METHOD(numInput);
	static NAN_METHOD(numOutput);
	static NAN_METHOD(getInput);
	static NAN_METHOD(getOutput);
	static NAN_METHOD(getTrainInput);
	static NAN_METHOD(getTrainOutput);
	static NAN_METHOD(setTrainData);
	static NAN_METHOD(getMaxInput);
	static NAN_METHOD(getMinInput);
	static NAN_METHOD(getMaxOutput);
	static NAN_METHOD(getMinOutput);

};

}

