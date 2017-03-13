#include "training-data.h"
#include <nan.h>
#include "fann-includes.h"
#include <iostream>
#include "utils.h"
#include "training-data.h"

namespace fanny {

class TDIOWorker : public Nan::AsyncWorker {
public:
	TrainingData *trainingData;
	std::string filename;
	bool isSave;
	bool isFixed;
	unsigned int decimalPoint;


	TDIOWorker(
		Nan::Callback *callback,
		v8::Local<v8::Object> tdHolder,
		std::string &_filename,
		bool _isSave,
		bool _isFixed,
		unsigned int _decimalPoint
	) : Nan::AsyncWorker(callback), filename(_filename),
		isSave(_isSave), isFixed(_isFixed),
		decimalPoint(_decimalPoint) {
		SaveToPersistent("tdHolder", tdHolder);
		trainingData = Nan::ObjectWrap::Unwrap<TrainingData>(tdHolder);
	}

	~TDIOWorker() {}

	void Execute() {
		if (!isSave) {
			if (!trainingData->trainingData->read_train_from_file(filename)) {
				SetErrorMessage("Error reading training data file");
			}
		} else if (!isFixed) {
			if (!trainingData->trainingData->save_train(filename)) {
				SetErrorMessage("Error saving training data file");
			}
		} else {
			if (!trainingData->trainingData->save_train_to_fixed(filename, decimalPoint)) {
				SetErrorMessage("Error saving training data file");
			}
		}
	}
};


Nan::Persistent<v8::FunctionTemplate> TrainingData::constructorFunctionTpl;

void TrainingData::Init(v8::Local<v8::Object> target) {
	// Create new function template for this JS class constructor
	v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
	// Set the class name
	tpl->SetClassName(Nan::New("TrainingData").ToLocalChecked());
	// Set the number of "slots" to allocate for fields on this class, not including prototype methods
	tpl->InstanceTemplate()->SetInternalFieldCount(1);
	// Save a reference to the function template
	TrainingData::constructorFunctionTpl.Reset(tpl);

	// Add prototype methods
	Nan::SetPrototypeMethod(tpl, "readTrainFromFile", readTrainFromFile);
	Nan::SetPrototypeMethod(tpl, "saveTrain", saveTrain);
	Nan::SetPrototypeMethod(tpl, "saveTrainToFixed", saveTrainToFixed);
	Nan::SetPrototypeMethod(tpl, "scaleInputTrainData", scaleInputTrainData);
	Nan::SetPrototypeMethod(tpl, "scaleOutputTrainData", scaleOutputTrainData);
	Nan::SetPrototypeMethod(tpl, "scaleTrainData", scaleTrainData);
	Nan::SetPrototypeMethod(tpl, "subsetTrainData", subsetTrainData);
	Nan::SetPrototypeMethod(tpl, "shuffle", shuffle);
	Nan::SetPrototypeMethod(tpl, "merge", merge);
	Nan::SetPrototypeMethod(tpl, "length", length);
	Nan::SetPrototypeMethod(tpl, "numInput", numInput);
	Nan::SetPrototypeMethod(tpl, "numOutput", numOutput);
	Nan::SetPrototypeMethod(tpl, "getInput", getInput);
	Nan::SetPrototypeMethod(tpl, "getOutput", getOutput);
	Nan::SetPrototypeMethod(tpl, "getTrainInput", getTrainInput);
	Nan::SetPrototypeMethod(tpl, "getTrainOutput", getTrainOutput);
	Nan::SetPrototypeMethod(tpl, "setTrainData", setTrainData);
	Nan::SetPrototypeMethod(tpl, "getMinInput", getMinInput);
	Nan::SetPrototypeMethod(tpl, "getMaxInput", getMaxInput);
	Nan::SetPrototypeMethod(tpl, "getMinOutput", getMinOutput);
	Nan::SetPrototypeMethod(tpl, "getMaxOutput", getMaxOutput);

	// Assign a property called 'TrainingData' to module.exports, pointing to our constructor
	Nan::Set(target, Nan::New("TrainingData").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

TrainingData::TrainingData(FANN::training_data *_training_data) : trainingData(_training_data) {}

TrainingData::~TrainingData() {
	delete trainingData;
}

NAN_METHOD(TrainingData::New) {
	FANN::training_data *trainingData;
	if (info.Length() == 1 && Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		TrainingData *other = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());
		FANN::training_data *otherTrainingData = other->trainingData;
		if (otherTrainingData->length_train_data() > 0) {
			trainingData = new FANN::training_data(*otherTrainingData);
		} else {
			trainingData = new FANN::training_data();
		}
	} else {
		trainingData = new FANN::training_data();
	}
	TrainingData *obj = new TrainingData(trainingData);
	obj->Wrap(info.This());
	info.GetReturnValue().Set(info.This());
}

NAN_METHOD(TrainingData::shuffle) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->shuffle_train_data();
}

NAN_METHOD(TrainingData::merge) {
	if (info.Length() != 1) {
		return Nan::ThrowError("Requires single argument");
	}
	if (!Nan::New(TrainingData::constructorFunctionTpl)->HasInstance(info[0])) {
		return Nan::ThrowError("Must be an instance of TrainingData");
	}
	TrainingData *other = Nan::ObjectWrap::Unwrap<TrainingData>(info[0].As<v8::Object>());
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->merge_train_data(*other->trainingData);
}

NAN_METHOD(TrainingData::length) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->length_train_data());
}

NAN_METHOD(TrainingData::numInput) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->num_input_train_data());
}

NAN_METHOD(TrainingData::numOutput) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->num_output_train_data());
}

NAN_METHOD(TrainingData::getInput) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	fann_type **data = self->trainingData->get_input();
	info.GetReturnValue().Set(fannDataSetToV8Array(data, self->trainingData->length_train_data(), self->trainingData->num_input_train_data()));
}

NAN_METHOD(TrainingData::getOutput) {
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	fann_type **data = self->trainingData->get_output();
	info.GetReturnValue().Set(fannDataSetToV8Array(data, self->trainingData->length_train_data(), self->trainingData->num_output_train_data()));
}

NAN_METHOD(TrainingData::getTrainInput) {
	if (info.Length() != 1 || !info[0]->IsNumber()) {
		return Nan::ThrowError("Argument must be a number");
	}
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	unsigned int pos = info[0]->Uint32Value();
	fann_type *data = self->trainingData->get_train_input(pos);
	info.GetReturnValue().Set(fannDataToV8Array(data, self->trainingData->num_input_train_data()));
}

NAN_METHOD(TrainingData::getTrainOutput) {
	if (info.Length() != 1 || !info[0]->IsNumber()) {
		return Nan::ThrowError("Argument must be a number");
	}
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	unsigned int pos = info[0]->Uint32Value();
	fann_type *data = self->trainingData->get_train_output(pos);
	info.GetReturnValue().Set(fannDataToV8Array(data, self->trainingData->num_output_train_data()));
}

NAN_METHOD(TrainingData::setTrainData) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: input, output");
	if (!info[0]->IsArray() || !info[1]->IsArray()) return Nan::ThrowError("Not an array");
	v8::Local<v8::Array> inputs = info[0].As<v8::Array>();
	v8::Local<v8::Array> outputs = info[1].As<v8::Array>();
	unsigned int dataSetLength = inputs->Length();
	if (outputs->Length() != dataSetLength) return Nan::ThrowError("Input and output dataset sizes must match");
	if (!dataSetLength) return Nan::ThrowError("Dataset must be nonzero in size");
	std::vector<fann_type> inputVector, outputVector;
	unsigned int numInputNodes = 0, numOutputNodes = 0;
	for (unsigned int idx = 0; idx < dataSetLength; ++idx) {
		Nan::MaybeLocal<v8::Value> inputMaybeValue = Nan::Get(inputs, idx);
		Nan::MaybeLocal<v8::Value> outputMaybeValue = Nan::Get(outputs, idx);
		if (inputMaybeValue.IsEmpty() || outputMaybeValue.IsEmpty()) return Nan::ThrowError("Invalid data");
		v8::Local<v8::Value> inputValue = inputMaybeValue.ToLocalChecked();
		v8::Local<v8::Value> outputValue = outputMaybeValue.ToLocalChecked();
		if (!inputValue->IsArray() || !outputValue->IsArray()) return Nan::ThrowError("Invalid data");
		v8::Local<v8::Array> inputArray = inputValue.As<v8::Array>();
		v8::Local<v8::Array> outputArray = outputValue.As<v8::Array>();
		if (idx == 0) {
			numInputNodes = inputArray->Length();
			numOutputNodes = outputArray->Length();
			if (!numInputNodes || !numOutputNodes) return Nan::ThrowError("Invalid data");
			inputVector.reserve(dataSetLength * numInputNodes);
			outputVector.reserve(dataSetLength * numOutputNodes);
		}
		std::vector<fann_type> inputRow = v8ArrayToFannData(inputArray);
		std::vector<fann_type> outputRow = v8ArrayToFannData(outputArray);
		if (inputRow.size() != numInputNodes || outputRow.size() != numOutputNodes) return Nan::ThrowError("Invalid data");
		memcpy(&inputVector[idx * numInputNodes], &inputRow[0], numInputNodes * sizeof(fann_type));
		memcpy(&outputVector[idx * numOutputNodes], &outputRow[0], numOutputNodes * sizeof(fann_type));
	}
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->set_train_data(dataSetLength, numInputNodes, &inputVector[0], numOutputNodes, &outputVector[0]);
}

NAN_METHOD(TrainingData::getMinInput) {
	#ifndef FANNY_FIXED
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->get_min_input());
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(TrainingData::getMaxInput) {
	#ifndef FANNY_FIXED
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->get_max_input());
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(TrainingData::getMinOutput) {
	#ifndef FANNY_FIXED
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->get_min_output());
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(TrainingData::getMaxOutput) {
	#ifndef FANNY_FIXED
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	info.GetReturnValue().Set(self->trainingData->get_max_output());
	#else
	Nan::ThrowError("Not supported for fixed fann");
	#endif
}

NAN_METHOD(TrainingData::readTrainFromFile) {
	if (info.Length() < 2 || !info[0]->IsString()) return Nan::ThrowError("Filename required");
	std::string filename = std::string(*v8::String::Utf8Value(info[0]));
	Nan::Callback *callback = new Nan::Callback(info[1].As<v8::Function>());
	AsyncQueueWorker(new TDIOWorker(callback, info.Holder(), filename, false, false, 0));
}

NAN_METHOD(TrainingData::saveTrain) {
	if (info.Length() < 2 || !info[0]->IsString()) return Nan::ThrowError("Filename required");
	std::string filename = std::string(*v8::String::Utf8Value(info[0]));
	Nan::Callback *callback = new Nan::Callback(info[1].As<v8::Function>());
	AsyncQueueWorker(new TDIOWorker(callback, info.Holder(), filename, true, false, 0));
}

NAN_METHOD(TrainingData::saveTrainToFixed) {
	if (info.Length() < 3 || !info[0]->IsString() || !info[1]->IsNumber()) return Nan::ThrowError("Filename and decimalPoint required");
	std::string filename = std::string(*v8::String::Utf8Value(info[0]));
	unsigned int decimalPoint = info[1]->Uint32Value();
	Nan::Callback *callback = new Nan::Callback(info[1].As<v8::Function>());
	AsyncQueueWorker(new TDIOWorker(callback, info.Holder(), filename, true, true, decimalPoint));
}

NAN_METHOD(TrainingData::scaleInputTrainData) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: new_min, new_max");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) return Nan::ThrowError("Arguments must be numbers");

	fann_type newMin = v8NumberToFannType(info[0]);
	fann_type newMax = v8NumberToFannType(info[1]);

	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->scale_input_train_data(newMin, newMax);
}

NAN_METHOD(TrainingData::scaleOutputTrainData) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: new_min, new_max");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) return Nan::ThrowError("Arguments must be numbers");

	fann_type newMin = v8NumberToFannType(info[0]);
	fann_type newMax = v8NumberToFannType(info[1]);

	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->scale_output_train_data(newMin, newMax);
}

NAN_METHOD(TrainingData::scaleTrainData) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: new_min, new_max");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) return Nan::ThrowError("Arguments must be numbers");

	fann_type newMin = v8NumberToFannType(info[0]);
	fann_type newMax = v8NumberToFannType(info[1]);

	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->scale_train_data(newMin, newMax);
}

NAN_METHOD(TrainingData::subsetTrainData) {
	if (info.Length() != 2) return Nan::ThrowError("Must have 2 arguments: pos, legth");
	if (!info[0]->IsNumber() || !info[1]->IsNumber()) return Nan::ThrowError("Arguments must be numbers");

	unsigned int pos = info[0]->Uint32Value();
	unsigned int length = info[1]->Uint32Value();

	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	self->trainingData->subset_train_data(pos, length);
}

}
