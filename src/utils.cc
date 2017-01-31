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

bool v8StringToStopFunctionEnum(v8::Local<v8::Value> value, FANN::stop_function_enum &ret) {
	if (!value->IsString()) return false;
	std::string str(*v8::String::Utf8Value(value));
	if (str.compare("STOPFUNC_MSE") == 0) ret = FANN::STOPFUNC_MSE;
	else if (str.compare("STOPFUNC_BIT") == 0) ret = FANN::STOPFUNC_BIT;
	else return false;
	return true;
}

v8::Local<v8::Object> connectionToV8Object(FANN::connection connection) {
	v8::Local<v8::Object> conncetionObject = Nan::New<v8::Object>();
	v8::Local<v8::Value> fromNeuron = Nan::New<v8::Number>(connection.from_neuron);
	v8::Local<v8::Value> toNeuron = Nan::New<v8::Number>(connection.to_neuron);
	v8::Local<v8::Value> weight = Nan::New<v8::Number>(connection.weight);

	Nan::Set(conncetionObject, Nan::New<v8::String>("from_neuron").ToLocalChecked(), fromNeuron);
	Nan::Set(conncetionObject, Nan::New<v8::String>("to_neuron").ToLocalChecked(), toNeuron);
	Nan::Set(conncetionObject, Nan::New<v8::String>("weight").ToLocalChecked(), weight);

	return conncetionObject;
}

v8::Local<v8::Value> connectionArrayToToV8Array(std::vector<FANN::connection> connectionArray, unsigned int size) {
	Nan::EscapableHandleScope scope;
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		v8::Local<v8::Object> value = connectionToV8Object(connectionArray[idx]);
		Nan::Set(v8Array, idx, value);
	}
	return scope.Escape(v8Array);
}

v8::Local<v8::Value> activationFunctionEnumToV8String(FANN::activation_function_enum value) {
	Nan::EscapableHandleScope scope;
	const char *str = NULL;
	fann_activationfunc_enum cValue = *(reinterpret_cast<fann_activationfunc_enum *>(&value));
	switch(cValue) {
		case FANN_LINEAR: str = "FANN_LINEAR"; break;
		case FANN_THRESHOLD: str = "FANN_THRESHOLD"; break;
		case FANN_THRESHOLD_SYMMETRIC: str = "FANN_THRESHOLD_SYMMETRIC"; break;
		case FANN_SIGMOID: str = "FANN_SIGMOID"; break;
		case FANN_SIGMOID_STEPWISE: str = "FANN_SIGMOID_STEPWISE"; break;
		case FANN_SIGMOID_SYMMETRIC: str = "FANN_SIGMOID_SYMMETRIC"; break;
		case FANN_SIGMOID_SYMMETRIC_STEPWISE: str = "FANN_SIGMOID_SYMMETRIC_STEPWISE"; break;
		case FANN_GAUSSIAN: str = "FANN_GAUSSIAN"; break;
		case FANN_GAUSSIAN_SYMMETRIC: str = "FANN_GAUSSIAN_SYMMETRIC"; break;
		case FANN_GAUSSIAN_STEPWISE: str = "FANN_GAUSSIAN_STEPWISE"; break;
		case FANN_ELLIOT: str = "FANN_ELLIOT"; break;
		case FANN_ELLIOT_SYMMETRIC: str = "FANN_ELLIOT_SYMMETRIC"; break;
		case FANN_LINEAR_PIECE: str = "FANN_LINEAR_PIECE"; break;
		case FANN_LINEAR_PIECE_SYMMETRIC: str = "FANN_LINEAR_PIECE_SYMMETRIC"; break;
		case FANN_SIN_SYMMETRIC: str = "FANN_SIN_SYMMETRIC"; break;
		case FANN_COS_SYMMETRIC: str = "FANN_COS_SYMMETRIC"; break;
		case FANN_COS: str = "FANN_COS"; break;
		case FANN_SIN: str = "FANN_SIN"; break;
	}
	v8::Local<v8::Value> ret;
	if (str) {
		ret = Nan::New<v8::String>(str).ToLocalChecked();
	} else {
		ret = Nan::Null();
	}
	return scope.Escape(ret);
}

bool v8StringToActivationFunctionEnum(v8::Local<v8::Value> value, FANN::activation_function_enum &activation_function) {
	if (!value->IsString()) return false;
	std::string str(*v8::String::Utf8Value(value));
	fann_activationfunc_enum ret;
	if (str.compare("FANN_LINEAR") == 0) ret = FANN_LINEAR;
	else if (str.compare("FANN_THRESHOLD") == 0) ret = FANN_THRESHOLD;
	else if (str.compare("FANN_THRESHOLD_SYMMETRIC") == 0) ret = FANN_THRESHOLD_SYMMETRIC;
	else if (str.compare("FANN_SIGMOID") == 0) ret = FANN_SIGMOID;
	else if (str.compare("FANN_SIGMOID_STEPWISE") == 0) ret = FANN_SIGMOID_STEPWISE;
	else if (str.compare("FANN_SIGMOID_SYMMETRIC") == 0) ret = FANN_SIGMOID_SYMMETRIC;
	else if (str.compare("FANN_SIGMOID_SYMMETRIC_STEPWISE") == 0) ret = FANN_SIGMOID_SYMMETRIC_STEPWISE;
	else if (str.compare("FANN_GAUSSIAN") == 0) ret = FANN_GAUSSIAN;
	else if (str.compare("FANN_GAUSSIAN_SYMMETRIC") == 0) ret = FANN_GAUSSIAN_SYMMETRIC;
	else if (str.compare("FANN_GAUSSIAN_STEPWISE") == 0) ret = FANN_GAUSSIAN_STEPWISE;
	else if (str.compare("FANN_ELLIOT") == 0) ret = FANN_ELLIOT;
	else if (str.compare("FANN_ELLIOT_SYMMETRIC") == 0) ret = FANN_ELLIOT_SYMMETRIC;
	else if (str.compare("FANN_LINEAR_PIECE") == 0) ret = FANN_LINEAR_PIECE;
	else if (str.compare("FANN_LINEAR_PIECE_SYMMETRIC") == 0) ret = FANN_LINEAR_PIECE_SYMMETRIC;
	else if (str.compare("FANN_SIN_SYMMETRIC") == 0) ret = FANN_SIN_SYMMETRIC;
	else if (str.compare("FANN_COS_SYMMETRIC") == 0) ret = FANN_COS_SYMMETRIC;
	else if (str.compare("FANN_COS") == 0) ret = FANN_COS;
	else if (str.compare("FANN_SIN") == 0) ret = FANN_SIN;
	else return false;

	activation_function = *(reinterpret_cast<FANN::activation_function_enum *>(&ret));
	return true;
}


}
