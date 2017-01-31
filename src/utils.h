#include "fann-includes.h"
#include <nan.h>
#include <vector>

namespace fanny {

std::vector<fann_type> v8ArrayToFannData(v8::Local<v8::Value> v8Array);

v8::Local<v8::Value> fannDataToV8Array(fann_type * data, unsigned int size);

v8::Local<v8::Value> fannDataSetToV8Array(fann_type ** data, unsigned int length, unsigned int size);

fann_type v8NumberToFannType(v8::Local<v8::Value> number);

v8::Local<v8::Value> trainingAlgorithmEnumToV8String(FANN::training_algorithm_enum * value);

bool v8StringToTrainingAlgorithmEnum(v8::Local<v8::Value> value, FANN::training_algorithm_enum &ret);

v8::Local<v8::Object> connectionToV8Object(FANN::connection connection);

v8::Local<v8::Value> connectionArrayToToV8Array(std::vector<FANN::connection> connectionArray, unsigned int size);
}
