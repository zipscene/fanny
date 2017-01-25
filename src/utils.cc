#include "utils.h"

namespace fanny {

std::vector<fann_type> v8ArrayToFannData(v8::Local<v8::Value> v8Array) {
	std::vector<fann_type> result;
	if (v8Array->IsArray()) {
		v8::Local<v8::Array> localArray = v8Array.As<v8::Array>();
		uint32_t length = localArray->Length();
		for (uint32_t idx = 0; idx < length; ++idx) {
			Nan::MaybeLocal<v8::Value> maybeIdxValue = Nan::Get(localArray, idx);
			if (!maybeIdxValue.IsEmpty()) {
				v8::Local<v8::Value> value = maybeIdxValue.ToLocalChecked();
				if (value->IsNumber()) {
					#ifdef FANNY_FIXED
					result.push_back(value->Int32Value());
					#else
					result.push_back(value->NumberValue());
					#endif
				}
			}
		}
	}
	return result;
}

v8::Local<v8::Value> fannDataToV8Array(fann_type * data, unsigned int size) {
	Nan::EscapableHandleScope scope;
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		v8::Local<v8::Value> value = Nan::New<v8::Number>(data[idx]);
		Nan::Set(v8Array, idx, value);
	}
	return scope.Escape(v8Array);
}

// length is how many entries are in data, size is the number of fann_type values in each entry
v8::Local<v8::Value> fannDataSetToV8Array(fann_type ** data, unsigned int length, unsigned int size) {
	Nan::EscapableHandleScope scope;
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < length; ++idx) {
		v8::Local<v8::Value> value = fannDataToV8Array(data[idx], size);
		Nan::Set(v8Array, idx, value);
	}
	return scope.Escape(v8Array);
}

fann_type v8NumberToFannType(v8::Local<v8::Value> number) {
	fann_type fannNumber;
	if (number->IsNumber()) {
		#ifdef FANNY_FIXED
		fannNumber = number->Uint32Value();
		#else
		fannNumber = number->NumberValue();
		#endif
	}
}

// TODO :: still needs testing
v8::Local<v8::Value> trainingAlgorithmEnumToV8String(FANN::training_algorithm_enum value) {
	Nan::EscapableHandleScope scope;
	const char *str = NULL;
	switch(value) {
		case FANN::TRAIN_INCREMENTAL: str = "TRAIN_INCREMENTAL"; break;
		case FANN::TRAIN_BATCH: str = "TRAIN_BATCH"; break;
		case FANN::TRAIN_RPROP: str = "TRAIN_RPROP"; break;
		case FANN::TRAIN_QUICKPROP: str = "TRAIN_QUICKPROP"; break;
		case FANN::TRAIN_SARPROP: str = "FANN_TRAIN_SARPROP"; break;
	}
	v8::Local<v8::Value> ret;
	if (str) {
		ret = Nan::New<v8::String>(str).ToLocalChecked();
	} else {
		ret = Nan::Null();
	}
	return scope.Escape(ret);
}

// TODO :: still needs testing
bool v8StringToTrainingAlgorithmEnum(v8::Local<v8::Value> value, FANN::training_algorithm_enum &ret) {
	if (!value->IsString()) return false;
	std::string str(*v8::String::Utf8Value(value));
	if (str.compare("TRAIN_INCREMENTAL")) ret = FANN::TRAIN_INCREMENTAL;
	else if (str.compare("TRAIN_BATCH")) ret = FANN::TRAIN_BATCH;
	else if (str.compare("TRAIN_RPROP")) ret = FANN::TRAIN_RPROP;
	else if (str.compare("TRAIN_QUICKPROP")) ret = FANN::TRAIN_QUICKPROP;
	else if (str.compare("FANN_TRAIN_SARPROP")) ret = FANN::TRAIN_SARPROP;
	else return false;
	return true;
}

}

	return fannNumber;
}

}
