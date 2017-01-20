#include "fanny.h"
#include <nan.h>
#include <fann_cpp.h>
#include <iostream>

namespace fanny {

/*class TestWorker : public Nan::AsyncWorker {

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

};*/

void FANNY::Init(v8::Local<v8::Object> target) {
	// Create new function template for this JS class constructor
	v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
	// Set the class name
	tpl->SetClassName(Nan::New("FANNY").ToLocalChecked());
	// Set the number of "slots" to allocate for fields on this class, not including prototype methods
	tpl->InstanceTemplate()->SetInternalFieldCount(1);

	// Add prototype methods
	//Nan::SetPrototypeMethod(tpl, "run", run);
	//Nan::SetPrototypeMethod(tpl, "runSync", runSync);

	// Assign a property called 'FANNY' to module.exports, pointing to our constructor
	Nan::Set(target, Nan::New("FANNY").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

FANNY::FANNY(FANN::neural_net *_fann) : fann(_fann) {}

FANNY::~FANNY() {
	delete fann;
}

NAN_METHOD(FANNY::New) {
	// Ensure arguments
	if (info.Length() != 1) {
		return Nan::ThrowError("Requires options argument");
	}
	if (!info[0]->IsObject()) {
		return Nan::ThrowTypeError("Invalid argument type");
	}

	// Get the options argument
	v8::Local<v8::Object> optionsObj(info[0].As<v8::Object>());

	// Variables for individual options
	std::string optType;
	std::vector<unsigned int> optLayers;
	float optConnectionRate = 0.5;

	// Get the type option
	Nan::MaybeLocal<v8::Value> maybeType = Nan::Get(optionsObj, Nan::New("type").ToLocalChecked());
	if (!maybeType.IsEmpty()) {
		v8::Local<v8::Value> localType = maybeType.ToLocalChecked();
		if (localType->IsString()) {
			optType = std::string(*(v8::String::Utf8Value(localType)));
		}
	}

	// Get the layers option
	Nan::MaybeLocal<v8::Value> maybeLayers = Nan::Get(optionsObj, Nan::New("layers").ToLocalChecked());
	if (!maybeLayers.IsEmpty()) {
		v8::Local<v8::Value> localLayers = maybeLayers.ToLocalChecked();
		if (localLayers->IsArray()) {
			v8::Local<v8::Array> arrayLayers = localLayers.As<v8::Array>();
			uint32_t length = arrayLayers->Length();
			for (uint32_t idx = 0; idx < length; ++idx) {
				Nan::MaybeLocal<v8::Value> maybeIdxValue = Nan::Get(arrayLayers, idx);
				if (!maybeIdxValue.IsEmpty()) {
					v8::Local<v8::Value> localIdxValue = maybeIdxValue.ToLocalChecked();
					if (localIdxValue->IsNumber()) {
						unsigned int idxValue = localIdxValue->Uint32Value();
						optLayers.push_back(idxValue);
					}
				}
			}
		}
	}
	if (optLayers.size() < 2) {
		return Nan::ThrowError("layers option is required with at least 2 layers");
	}

	// Get the connectionRate option
	Nan::MaybeLocal<v8::Value> maybeConnectionRate = Nan::Get(optionsObj, Nan::New("connectionRate").ToLocalChecked());
	if (!maybeConnectionRate.IsEmpty()) {
		v8::Local<v8::Value> localConnectionRate = maybeConnectionRate.ToLocalChecked();
		if (localConnectionRate->IsNumber()) {
			optConnectionRate = localConnectionRate->NumberValue();
		}
	}

	// Construct the neural_net underlying class
	FANN::neural_net *fann;
	if (!optType.compare("standard") || optType.empty()) {
		fann = new FANN::neural_net(FANN::network_type_enum::LAYER, (unsigned int)optLayers.size(), (const unsigned int *)&optLayers[0]);
	} else if(optType.compare("sparse")) {
		fann = new FANN::neural_net(optConnectionRate, optLayers.size(), &optLayers[0]);
	} else if (optType.compare("shortcut")) {
		fann = new FANN::neural_net(FANN::network_type_enum::SHORTCUT, optLayers.size(), &optLayers[0]);
	} else {
		return Nan::ThrowError("Invalid type option");
	}

	FANNY *obj = new FANNY(fann);
	obj->Wrap(info.This());
	info.GetReturnValue().Set(info.This());
}
/*
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
*/

}

