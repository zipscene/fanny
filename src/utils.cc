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
	// TODO: Better error checking around this value being set
	fann_type fannNumber = 0;
	if (number->IsNumber()) {
		#ifdef FANNY_FIXED
		fannNumber = number->Uint32Value();
		#else
		fannNumber = number->NumberValue();
		#endif
	}
	return fannNumber;
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
	if (str.compare("TRAIN_INCREMENTAL") == 0) ret = FANN::TRAIN_INCREMENTAL;
	else if (str.compare("TRAIN_BATCH") == 0) ret = FANN::TRAIN_BATCH;
	else if (str.compare("TRAIN_RPROP") == 0) ret = FANN::TRAIN_RPROP;
	else if (str.compare("TRAIN_QUICKPROP") == 0) ret = FANN::TRAIN_QUICKPROP;
	else if (str.compare("FANN_TRAIN_SARPROP") == 0) ret = FANN::TRAIN_SARPROP;
	else return false;
	return true;
}

v8::Local<v8::Value> errorFunctionEnumToV8String(FANN::error_function_enum value) {
	Nan::EscapableHandleScope scope;
	const char *str = NULL;
	switch(value) {
		case FANN::ERRORFUNC_LINEAR: str = "ERRORFUNC_LINEAR"; break;
		case FANN::ERRORFUNC_TANH: str = "ERRORFUNC_TANH"; break;
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
bool v8StringToErrorFunctionEnum(v8::Local<v8::Value> value, FANN::error_function_enum &ret) {
	if (!value->IsString()) return false;
	std::string str(*v8::String::Utf8Value(value));
	if (str.compare("ERRORFUNC_LINEAR") == 0) ret = FANN::ERRORFUNC_LINEAR;
	else if (str.compare("ERRORFUNC_TANH") == 0) ret = FANN::ERRORFUNC_TANH;
	else return false;
	return true;
}

v8::Local<v8::Value> stopFunctionEnumToV8String(FANN::stop_function_enum value) {
	Nan::EscapableHandleScope scope;
	const char *str = NULL;
	switch(value) {
		case FANN::STOPFUNC_MSE: str = "STOPFUNC_MSE"; break;
		case FANN::STOPFUNC_BIT: str = "STOPFUNC_BIT"; break;
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
bool v8StringToStopFunctionEnum(v8::Local<v8::Value> value, FANN::stop_function_enum &ret) {
	if (!value->IsString()) return false;
	std::string str(*v8::String::Utf8Value(value));
	if (str.compare("STOPFUNC_MSE") == 0) ret = FANN::STOPFUNC_MSE;
	else if (str.compare("STOPFUNC_BIT") == 0) ret = FANN::STOPFUNC_BIT;
	else return false;
	return true;
}

}
