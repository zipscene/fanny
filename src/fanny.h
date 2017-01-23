#include <nan.h>
#include "fann-includes.h"

namespace fanny {

class FANNY : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static void Init(v8::Local<v8::Object> target);

private:

	// Javascript Constructor.  Takes single "options" object parameter.  Options can include:
	// - type (string) - One of "standard", "sparse", "shortcut"
	// - layers (array of numbers)
	// - connectionRate (number) - For sparse networks
	static NAN_METHOD(New);

	// FANN "run" method.  Parameter is array of numbers.  Returns array of numbers.
	// Also takes a callback.
	//static NAN_METHOD(runAsync);

	// Synchronous version of "run".
	static NAN_METHOD(run);
	static NAN_METHOD(getNumInput);
	static NAN_METHOD(getNumOutput);
	static NAN_METHOD(getTotalNeurons);
	static NAN_METHOD(getTotalConnections);
	static NAN_METHOD(getNumLayers);
	static NAN_METHOD(getBitFail);
	static NAN_METHOD(getErrno);
	static NAN_METHOD(getMSE);
	static NAN_METHOD(getLearningRate);
	static NAN_METHOD(getQuickPropDecay);
	static NAN_METHOD(getQuickPropMu);
	static NAN_METHOD(getRpropIncreaseFactor);
	static NAN_METHOD(getRpropDecreaseFactor);
	static NAN_METHOD(getRpropDeltaZero);
	static NAN_METHOD(getRpropDeltaMin);
	static NAN_METHOD(getRpropDeltaMax);

	// constructor & destructor
	explicit FANNY(FANN::neural_net *fann);
	~FANNY();

	// Checks to see if the fann instance has a recorded error.
	// If so, this throws a JS error and returns true.
	// In this case, the calling function must return immediately.
	// It also resets the fann error.
	bool checkError();

	// Encapsulated FANN neural_net instance
	FANN::neural_net *fann;

};

}

