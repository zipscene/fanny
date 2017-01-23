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
	//static NAN_METHOD(clone);

};

}

