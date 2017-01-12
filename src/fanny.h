#include <nan.h>

namespace fanny {

class FANN : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static NAN_METHOD(Init) {
		// Create new function template for this JS class constructor
		v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
		// Set the class name
		tpl->SetClassName(Nan::New("FANN").ToLocalChecked());
		// Set the number of "slots" to allocate for fields on this class, not including prototype methods
		tpl->InstanceTemplate()->SetInternalFieldCount(0);

		// ???
		constructor().Reset(Nan::GetFunction(tpl).ToLocalChecked());
	}

private:

	static NAN_METHOD(New) {

	}

	// Function that returns the single, static constructor Function
	static inline Nan::Persistent<v8::Function> & constructor() {
		static Nan::Persistent<v8::Function> c;
		return c;
	}

}

}
