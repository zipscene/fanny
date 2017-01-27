#include "fann-includes.h"
#include <nan.h>
#include <vector>

namespace fanny {

std::vector<fann_type> v8ArrayToFannData(v8::Local<v8::Value> v8Array);

v8::Local<v8::Value> fannDataToV8Array(fann_type * data, unsigned int size);

v8::Local<v8::Value> fannDataSetToV8Array(fann_type ** data, unsigned int length, unsigned int size);

// Number converter
fann_type v8NumberToFannType(v8::Local<v8::Value> number);

// training_algorithm_enum converters
v8::Local<v8::Value> trainingAlgorithmEnumToV8String(FANN::training_algorithm_enum value);

bool v8StringToTrainingAlgorithmEnum(v8::Local<v8::Value> value, FANN::training_algorithm_enum &ret);

// error_function_enum converters
v8::Local<v8::Value> errorFunctionEnumToV8String(FANN::error_function_enum value);

bool v8StringToErrorFunctionEnum(v8::Local<v8::Value> value, FANN::error_function_enum &ret);

// stop_function_enum converters
v8::Local<v8::Value> stopFunctionEnumToV8String(FANN::stop_function_enum value);

bool v8StringToStopFunctionEnum(v8::Local<v8::Value> value, FANN::stop_function_enum &ret);
}
