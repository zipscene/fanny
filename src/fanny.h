#include <nan.h>
#include <floatfann.h>
#include <fann_cpp.h>

namespace fanny {

class FANNY : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static void Init(v8::Local<v8::Object> target);

private:

	// Neural network instance this class wraps
	FANN::neural_net *neural_net;

	// Javascript Constructor.  Takes single "options" object parameter.  Options can include:
	// - type (string) - One of "standard", "sparse", "shortcut"
	// - layers (array of numbers)
	// - connectionRate (number) - For sparse networks
	static NAN_METHOD(New);

	// FANN "run" method.  Parameter is array of numbers.  Returns array of numbers.
	// Also takes a callback.
	//static NAN_METHOD(run);

	// Synchronous version of "run".
	//static NAN_METHOD(runSync);

	// constructor & destructor
	explicit FANNY(FANN::neural_net *fann);
	~FANNY();

	// Encapsulated FANN neural_net instance
	FANN::neural_net *fann;

};

}
