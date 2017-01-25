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

}


