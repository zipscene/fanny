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

