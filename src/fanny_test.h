#include <nan.h>

namespace fanny {

class FANN : public Nan::ObjectWrap {

public:
	// Initialize this class and add itself to the exports
	// This is NOT the Javascript class constructor method
	static void Init(v8::Local<v8::Object> target);

private:

	static NAN_METHOD(test);

	static NAN_METHOD(New);
	// explicit constructor & destructor
	explicit FANN() {}
	~FANN() {}

};

}
