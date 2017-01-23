#include "fanny.h"
#include <nan.h>
#include "fann-includes.h"
#include <iostream>
#include "utils.h"

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
	Nan::SetPrototypeMethod(tpl, "run", run);
	Nan::SetPrototypeMethod(tpl, "getNumInput", getNumInput);
	Nan::SetPrototypeMethod(tpl, "getNumOutput", getNumOutput);
	Nan::SetPrototypeMethod(tpl, "getTotalNeurons", getTotalNeurons);
	Nan::SetPrototypeMethod(tpl, "getTotalConnections", getTotalConnections);
	Nan::SetPrototypeMethod(tpl, "getNumLayers", getNumLayers);
	Nan::SetPrototypeMethod(tpl, "getBitFail", getBitFail);
	Nan::SetPrototypeMethod(tpl, "getErrno", getErrno);
	Nan::SetPrototypeMethod(tpl, "getMSE", getMSE);
	Nan::SetPrototypeMethod(tpl, "getLearningRate", getLearningRate);
	Nan::SetPrototypeMethod(tpl, "getQuickPropDecay", getQuickPropDecay);
	Nan::SetPrototypeMethod(tpl, "getQuickPropMu", getQuickPropMu);
	Nan::SetPrototypeMethod(tpl, "getRpropIncreaseFactor", getRpropIncreaseFactor);
	Nan::SetPrototypeMethod(tpl, "getRpropDecreaseFactor", getRpropDecreaseFactor);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaZero", getRpropDeltaZero);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaMin", getRpropDeltaMin);
	Nan::SetPrototypeMethod(tpl, "getRpropDeltaMax", getRpropDeltaMax);
	//Nan::SetPrototypeMethod(tpl, "runAsync", runAsync);

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

bool FANNY::checkError() {
	unsigned int fannerr = fann->get_errno();
	if (fannerr) {
		std::string errstr = fann->get_errstr();
		std::string msg = std::string("FANN error ") + std::to_string(fannerr) + ": " + errstr;
		Nan::ThrowError(msg.c_str());
		fann->reset_errno();
		fann->reset_errstr();
		return true;
	} else {
		return false;
	}
}

NAN_METHOD(FANNY::run) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 1) return Nan::ThrowError("Missing argument");
	if (!info[0]->IsArray()) return Nan::ThrowError("Must be array");
	std::vector<fann_type> inputs = v8ArrayToFannData(info[0]);
	if (inputs.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	fann_type *outputs = fanny->fann->run(&inputs[0]);
	if (fanny->checkError()) return;
	v8::Local<v8::Value> outputArray = fannDataToV8Array(outputs, fanny->fann->get_num_output());
	info.GetReturnValue().Set(outputArray);
}

NAN_METHOD(FANNY::getNumInput) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_input();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getNumOutput) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_output();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getTotalNeurons) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_total_neurons();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getTotalConnections) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_total_connections();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getNumLayers) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_num_layers();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getBitFail) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_bit_fail();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getErrno) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	unsigned int num = fanny->fann->get_errno();
	info.GetReturnValue().Set(num);
}

NAN_METHOD(FANNY::getMSE) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_MSE();
	info.GetReturnValue().Set(num);
}

// by default 0.7
NAN_METHOD(FANNY::getLearningRate) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_learning_rate();
	info.GetReturnValue().Set(num);
}

// by default -0.0001
NAN_METHOD(FANNY::getQuickPropDecay) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_quickprop_decay();
	info.GetReturnValue().Set(num);
}

// by default 1.75
NAN_METHOD(FANNY::getQuickPropMu) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_quickprop_mu();
	info.GetReturnValue().Set(num);	
}

// by default 1.2
NAN_METHOD(FANNY::getRpropIncreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_increase_factor();
	info.GetReturnValue().Set(num);
}

// by default 0.5
NAN_METHOD(FANNY::getRpropDecreaseFactor) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_decrease_factor();
	info.GetReturnValue().Set(num);
}

// by default 0.1
NAN_METHOD(FANNY::getRpropDeltaZero) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_zero();
	info.GetReturnValue().Set(num);
}

// by default 0.0
NAN_METHOD(FANNY::getRpropDeltaMin) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_min();
	info.GetReturnValue().Set(num);
}

// by default 50.0
NAN_METHOD(FANNY::getRpropDeltaMax) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	float num = fanny->fann->get_rprop_delta_max();
	info.GetReturnValue().Set(num);
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

