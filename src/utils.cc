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
	Nan::EscapableHandleScope scope;
	v8::Local<v8::Object> connectionObject = Nan::New<v8::Object>();
	v8::Local<v8::Value> fromNeuron = Nan::New<v8::Number>(connection.from_neuron);
	v8::Local<v8::Value> toNeuron = Nan::New<v8::Number>(connection.to_neuron);
	v8::Local<v8::Value> weight = Nan::New<v8::Number>(connection.weight);

	Nan::Set(connectionObject, Nan::New<v8::String>("fromNeuron").ToLocalChecked(), fromNeuron);
	Nan::Set(connectionObject, Nan::New<v8::String>("toNeuron").ToLocalChecked(), toNeuron);
	Nan::Set(connectionObject, Nan::New<v8::String>("weight").ToLocalChecked(), weight);

	return scope.Escape(connectionObject);
}

v8::Local<v8::Value> connectionArrayToV8Array(std::vector<FANN::connection> connectionArray) {
	Nan::EscapableHandleScope scope;
	unsigned int size = connectionArray.size();
	v8::Local<v8::Array> v8Array = Nan::New<v8::Array>(size);
	for (uint32_t idx = 0; idx < size; ++idx) {
		v8::Local<v8::Object> value = connectionToV8Object(connectionArray[idx]);
		Nan::Set(v8Array, idx, value);
	}
	return scope.Escape(v8Array);
}

std::vector<FANN::connection> v8ArrayToConnection(v8::Local<v8::Value> v8Array) {
	std::vector<FANN::connection> result;
	if (v8Array->IsArray()) {
		v8::Local<v8::Array> localArray = v8Array.As<v8::Array>();
		uint32_t length = localArray->Length();
		result.reserve(length);
		for (uint32_t idx = 0; idx < length; ++idx) {
			Nan::MaybeLocal<v8::Value> maybeIdxValue = Nan::Get(localArray, idx);
			if (!maybeIdxValue.IsEmpty()) {
				v8::Local<v8::Value> value = maybeIdxValue.ToLocalChecked();
				if (value->IsObject()) {
					v8::Local<v8::Object> obj = value.As<v8::Object>();
					FANN::connection connection;
					unsigned int count = 0;
					Nan::MaybeLocal<v8::Value> maybeToNeuron = Nan::Get(obj, Nan::New("toNeuron").ToLocalChecked());
					if (!maybeToNeuron.IsEmpty()) {
						++count;
						connection.to_neuron = maybeToNeuron.ToLocalChecked()->Uint32Value();
					}
					Nan::MaybeLocal<v8::Value> maybeFromNeuron = Nan::Get(obj, Nan::New("fromNeuron").ToLocalChecked());
					if (!maybeFromNeuron.IsEmpty()) {
						++count;
						connection.from_neuron = maybeFromNeuron.ToLocalChecked()->Uint32Value();
					}
					Nan::MaybeLocal<v8::Value> maybeWeight = Nan::Get(obj, Nan::New("weight").ToLocalChecked());
					if (!maybeWeight.IsEmpty()) {
						++count;
						connection.weight = v8NumberToFannType(maybeWeight.ToLocalChecked());
					}
					if (count == 3) {
						result.push_back(connection);
					}
				}
			}
		}
	}
	return result;
}

v8::Local<v8::Value> activationFunctionEnumToV8String(FANN::activation_function_enum value) {
	Nan::EscapableHandleScope scope;
	const char *str = NULL;
	switch(value) {
		case FANN::LINEAR: str = "LINEAR"; break;
		case FANN::THRESHOLD: str = "THRESHOLD"; break;
		case FANN::THRESHOLD_SYMMETRIC: str = "THRESHOLD_SYMMETRIC"; break;
		case FANN::SIGMOID: str = "SIGMOID"; break;
		case FANN::SIGMOID_STEPWISE: str = "SIGMOID_STEPWISE"; break;
		case FANN::SIGMOID_SYMMETRIC: str = "SIGMOID_SYMMETRIC"; break;
		case FANN::SIGMOID_SYMMETRIC_STEPWISE: str = "SIGMOID_SYMMETRIC_STEPWISE"; break;
		case FANN::GAUSSIAN: str = "GAUSSIAN"; break;
		case FANN::GAUSSIAN_SYMMETRIC: str = "GAUSSIAN_SYMMETRIC"; break;
		case FANN::GAUSSIAN_STEPWISE: str = "GAUSSIAN_STEPWISE"; break;
		case FANN::ELLIOT: str = "ELLIOT"; break;
		case FANN::ELLIOT_SYMMETRIC: str = "ELLIOT_SYMMETRIC"; break;
		case FANN::LINEAR_PIECE: str = "LINEAR_PIECE"; break;
		case FANN::LINEAR_PIECE_SYMMETRIC: str = "LINEAR_PIECE_SYMMETRIC"; break;
		case FANN::SIN_SYMMETRIC: str = "SIN_SYMMETRIC"; break;
		case FANN::COS_SYMMETRIC: str = "COS_SYMMETRIC"; break;
		case FANN::COS: str = "COS"; break;
		case FANN::SIN: str = "SIN"; break;
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
	FANN::activation_function_enum ret;
	if (str.compare("LINEAR") == 0) ret = FANN::LINEAR;
	else if (str.compare("THRESHOLD") == 0) ret = FANN::THRESHOLD;
	else if (str.compare("THRESHOLD_SYMMETRIC") == 0) ret = FANN::THRESHOLD_SYMMETRIC;
	else if (str.compare("SIGMOID") == 0) ret = FANN::SIGMOID;
	else if (str.compare("SIGMOID_STEPWISE") == 0) ret = FANN::SIGMOID_STEPWISE;
	else if (str.compare("SIGMOID_SYMMETRIC") == 0) ret = FANN::SIGMOID_SYMMETRIC;
	else if (str.compare("SIGMOID_SYMMETRIC_STEPWISE") == 0) ret = FANN::SIGMOID_SYMMETRIC_STEPWISE;
	else if (str.compare("GAUSSIAN") == 0) ret = FANN::GAUSSIAN;
	else if (str.compare("GAUSSIAN_SYMMETRIC") == 0) ret = FANN::GAUSSIAN_SYMMETRIC;
	else if (str.compare("GAUSSIAN_STEPWISE") == 0) ret = FANN::GAUSSIAN_STEPWISE;
	else if (str.compare("ELLIOT") == 0) ret = FANN::ELLIOT;
	else if (str.compare("ELLIOT_SYMMETRIC") == 0) ret = FANN::ELLIOT_SYMMETRIC;
	else if (str.compare("LINEAR_PIECE") == 0) ret = FANN::LINEAR_PIECE;
	else if (str.compare("LINEAR_PIECE_SYMMETRIC") == 0) ret = FANN::LINEAR_PIECE_SYMMETRIC;
	else if (str.compare("SIN_SYMMETRIC") == 0) ret = FANN::SIN_SYMMETRIC;
	else if (str.compare("COS_SYMMETRIC") == 0) ret = FANN::COS_SYMMETRIC;
	else if (str.compare("COS") == 0) ret = FANN::COS;
	else if (str.compare("SIN") == 0) ret = FANN::SIN;
	else return false;
	activation_function = ret;
	return true;
}


}
