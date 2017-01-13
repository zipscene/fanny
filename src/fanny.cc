#include "fanny.h"
#include <nan.h>
#include <iostream>

namespace fanny {

void FANN::Init(v8::Local<v8::Object> target) {
	// Create new function template for this JS class constructor
	v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
	// Set the class name
	tpl->SetClassName(Nan::New("FANN").ToLocalChecked());
	// Set the number of "slots" to allocate for fields on this class, not including prototype methods
	tpl->InstanceTemplate()->SetInternalFieldCount(1);

	// Add prototype methods
	Nan::SetPrototypeMethod(tpl, "test", test);

	// Assign a property called 'FANN' to module.exports, pointing to our constructor
	Nan::Set(target, Nan::New("FANN").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

NAN_METHOD(FANN::New) {
	FANN *obj = new FANN();
	obj->Wrap(info.This());
	info.GetReturnValue().Set(info.This());
}

NAN_METHOD(FANN::test) {
	std::cout << "Test!\n";
}

}

