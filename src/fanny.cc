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
				//v8::Local<v8::String> str = Nan::To<v8::String>(value).ToLocalChecked();
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
	std::cout << "Test!\n";
}

}

