#include "fanny.h"
#include <nan.h>
#include "fann-includes.h"
#include <iostream>
#include "utils.h"

namespace fanny {

class RunWorker : public Nan::AsyncWorker {

public:
	RunWorker(Nan::Callback *callback, std::vector<fann_type> & _inputs, v8::Local<v8::Object> fannyHolder) : Nan::AsyncWorker(callback), inputs(_inputs) {
		SaveToPersistent("fannyHolder", fannyHolder);
		fanny = Nan::ObjectWrap::Unwrap<FANNY>(fannyHolder);
	}
	~RunWorker() {}

	void Execute() {
		fann_type *fannOutputs = fanny->fann->run(&inputs[0]);
		if (fanny->fann->get_errno()) {
			SetErrorMessage(fanny->fann->get_errstr().c_str());
			fanny->fann->reset_errno();
			fanny->fann->reset_errstr();
			return;
		}
		unsigned int numOutputs = fanny->fann->get_num_output();
		for (unsigned int idx = 0; idx < numOutputs; idx++) {
			outputs.push_back(fannOutputs[idx]);
		}
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		v8::Local<v8::Value> args[] = {
			Nan::Null(),
			fannDataToV8Array(&outputs[0], outputs.size())
		};
		callback->Call(2, args);
	}

	std::vector<fann_type> inputs;
	std::vector<fann_type> outputs;
	FANNY *fanny;
};

class LoadFileWorker : public Nan::AsyncWorker {
public:
	LoadFileWorker(Nan::Callback *callback, std::string _filename) : Nan::AsyncWorker(callback), filename(_filename) {}
	~LoadFileWorker() {}

	void Execute() {
		struct fann *ann = fann_create_from_file(filename.c_str());
		if (!ann) return SetErrorMessage("Error loading FANN file");
		fann = new FANN::neural_net(ann);
		fann_destroy(ann);
	}

	void HandleOKCallback() {
		Nan::HandleScope scope;
		if (fann->get_errno()) {
			v8::Local<v8::Value> args[] = { Nan::Error(std::string(fann->get_errstr()).c_str()) };
			delete fann;
			callback->Call(1, args);
		} else {
			v8::Local<v8::Value> externFann = Nan::New<v8::External>(fann);
			v8::Local<v8::Function> ctor = Nan::New(FANNY::constructorFunction);
			v8::Local<v8::Value> ctorArgs[] = { externFann };
			v8::Local<v8::Value> cbargs[] = {
				Nan::Null(),
				Nan::NewInstance(ctor, 1, ctorArgs).ToLocalChecked()
			};
			callback->Call(2, cbargs);
		}
	}

	std::string filename;
	FANN::neural_net *fann;
};

void FANNY::Init(v8::Local<v8::Object> target) {
	// Create new function template for this JS class constructor
	v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
	// Set the class name
	tpl->SetClassName(Nan::New("FANNY").ToLocalChecked());
	// Set the number of "slots" to allocate for fields on this class, not including prototype methods
	tpl->InstanceTemplate()->SetInternalFieldCount(1);
	// Save a constructor reference
	FANNY::constructorFunctionTpl.Reset(tpl);

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
	Nan::SetPrototypeMethod(tpl, "runAsync", runAsync);

	// Create the loadFile function
	v8::Local<v8::FunctionTemplate> loadFileTpl = Nan::New<v8::FunctionTemplate>(loadFile);
	v8::Local<v8::Function> loadFileFunction = Nan::GetFunction(loadFileTpl).ToLocalChecked();

	// Assign a property called 'FANNY' to module.exports, pointing to our constructor
	v8::Local<v8::Function> ctorFunction = Nan::GetFunction(tpl).ToLocalChecked();
	Nan::Set(ctorFunction, Nan::New("loadFile").ToLocalChecked(), loadFileFunction);
	FANNY::constructorFunction.Reset(ctorFunction);
	Nan::Set(target, Nan::New("FANNY").ToLocalChecked(), ctorFunction);
}


Nan::Persistent<v8::FunctionTemplate> FANNY::constructorFunctionTpl;
Nan::Persistent<v8::Function> FANNY::constructorFunction;

FANNY::FANNY(FANN::neural_net *_fann) : fann(_fann) {}

FANNY::~FANNY() {
	delete fann;
	constructorFunctionTpl.Empty();
}

NAN_METHOD(FANNY::loadFile) {
	if (info.Length() != 2) return Nan::ThrowError("Requires filename and callback");
	std::string filename = *v8::String::Utf8Value(info[0]);
	Nan::Callback * callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new LoadFileWorker(callback, filename));
}

NAN_METHOD(FANNY::New) {
	// Ensure arguments
	if (info.Length() != 1) {
		return Nan::ThrowError("Requires single argument");
	}

	FANN::neural_net *fann;

	if (Nan::New(FANNY::constructorFunctionTpl)->HasInstance(info[0])) {
		// Copy constructor
		FANNY *other = Nan::ObjectWrap::Unwrap<FANNY>(info[0].As<v8::Object>());
		fann = new FANN::neural_net(*other->fann);
	} else if (info[0]->IsString()) {
		// Load-from-file constructor
		fann = new FANN::neural_net(std::string(*v8::String::Utf8Value(info[0])));
	} else if (info[0]->IsExternal()) {
		// Internal from-instance constructor
		fann = (FANN::neural_net *)info[0].As<v8::External>()->Value();
	} else if (info[0]->IsObject()) {
		// Options constructor

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
		if (!optType.compare("standard") || optType.empty()) {
			fann = new FANN::neural_net(FANN::network_type_enum::LAYER, (unsigned int)optLayers.size(), (const unsigned int *)&optLayers[0]);
		} else if(optType.compare("sparse")) {
			fann = new FANN::neural_net(optConnectionRate, optLayers.size(), &optLayers[0]);
		} else if (optType.compare("shortcut")) {
			fann = new FANN::neural_net(FANN::network_type_enum::SHORTCUT, optLayers.size(), &optLayers[0]);
		} else {
			return Nan::ThrowError("Invalid type option");
		}
	} else {
		return Nan::ThrowTypeError("Invalid argument type");
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
	if (info.Length() != 1) return Nan::ThrowError("Takes one argument");
	if (!info[0]->IsArray()) return Nan::ThrowError("Must be array");
	std::vector<fann_type> inputs = v8ArrayToFannData(info[0]);
	if (inputs.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	fann_type *outputs = fanny->fann->run(&inputs[0]);
	if (fanny->checkError()) return;
	v8::Local<v8::Value> outputArray = fannDataToV8Array(outputs, fanny->fann->get_num_output());
	info.GetReturnValue().Set(outputArray);
}

NAN_METHOD(FANNY::runAsync) {
	FANNY *fanny = Nan::ObjectWrap::Unwrap<FANNY>(info.Holder());
	if (info.Length() != 2) return Nan::ThrowError("Takes two arguments");
	if (!info[0]->IsArray()) return Nan::ThrowError("First argument must be array");
	if (!info[1]->IsFunction()) return Nan::ThrowError("Second argument must be callback");
	std::vector<fann_type> inputs = v8ArrayToFannData(info[0]);
	if (inputs.size() != fanny->fann->get_num_input()) return Nan::ThrowError("Wrong number of inputs");
	Nan::Callback * callback = new Nan::Callback(info[1].As<v8::Function>());
	Nan::AsyncQueueWorker(new RunWorker(callback, inputs, info.Holder()));
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

}
