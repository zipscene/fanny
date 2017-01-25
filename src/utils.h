#include "fann-includes.h"
#include <nan.h>
#include <vector>

namespace fanny {

std::vector<fann_type> v8ArrayToFannData(v8::Local<v8::Value> v8Array);

v8::Local<v8::Value> fannDataToV8Array(fann_type * data, unsigned int size);

v8::Local<v8::Value> fannDataSetToV8Array(fann_type ** data, unsigned int length, unsigned int size);

fann_type v8NumberToFannType(v8::Local<v8::Value> number);

}
