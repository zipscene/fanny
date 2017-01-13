#include "fanny.h"
#include <nan.h>
#include <iostream>

namespace fanny {

class TestWorker : public Nan::AsyncWorker {

public:
	TestWorker(Nan::Callback * callback) : Nan::AsyncWorker(callback) {}
	~TestWorker() {}

	void Execute() {
		std::cout << "Async!\n";
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		v8::Local<v8::Value> args[] = {
			Nan::Null(),
			Nan::New<v8::Number>(42)
		};
		callback->Call(2, args);
	}

};

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
	// Print out comma-separated array given as first arg
	uint32_t numArgs = info.Length();
	if (numArgs == 0) {
		std::cout << "No args!\n";
	} else if (info[0]->IsUndefined()) {
		std::cout << "Undefined!\n";
	} else if (info[0]->IsNull()) {
		std::cout << "Null!\n";
	} else if (info[0]->IsArray()) {
		v8::Local<v8::Array> array = v8::Local<v8::Array>::Cast(info[0]);
		for (uint32_t i = 0; i < array->Length(); i++) {
			v8::Local<v8::Value> value = array->Get(i);
			if (value->IsString()) {
				v8::String::Utf8Value utf8Str(value);
				const char * cStr = *utf8Str;
				std::cout << cStr << ", ";
			} else {
				std::cout << "<NotString>, ";
			}
		}
		std::cout << "\n";
	} else {
		std::cout << "Unrecognized type\n";
	}

	// Print out the string "Test"
	std::cout << "Test!\n";

	// Return an array we construct
	v8::Local<v8::Array> retArray = Nan::New<v8::Array>();
	retArray->Set(0, Nan::New<v8::String>("first").ToLocalChecked());
	retArray->Set(1, Nan::New<v8::String>("second").ToLocalChecked());
	info.GetReturnValue().Set(retArray);

	// If a callback is given, execute the async code
	if (numArgs >= 2 && info[1]->IsFunction()) {
		std::cout << "Starting async worker\n";
		Nan::Callback * callback = new Nan::Callback(info[1].As<v8::Function>());
		Nan::AsyncQueueWorker(new TestWorker(callback));
	}
}

}

