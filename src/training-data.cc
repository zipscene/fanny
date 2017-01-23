#include "training-data.h"
#include <nan.h>
#include "fann-includes.h"
#include <iostream>
#include "utils.h"

namespace fanny {

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
	//Nan::SetPrototypeMethod(tpl, "clone", clone);

	// Assign a property called 'TrainingData' to module.exports, pointing to our constructor
	Nan::Set(target, Nan::New("TrainingData").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

TrainingData::TrainingData(FANN::training_data *_training_data) : trainingData(_training_data) {}

TrainingData::~TrainingData() {
	delete trainingData;
	constructorFunctionTpl.Empty();
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

/*NAN_METHOD(TrainingData::clone) {
	Nan::EscapableHandleScope scope;
	TrainingData *self = Nan::ObjectWrap::Unwrap<TrainingData>(info.Holder());
	TrainingData *obj = new TrainingData(new FANN::training_data(*self->trainingData));
	v8::Local<v8::Object>
	obj->Wrap()
}*/


}

