#include <floatfann.h>
#include <fann_cpp.h>
#include <nan.h>
#include <vector>

namespace fanny {

std::vector<FANN::fann_type> v8ArrayToFannData(v8::Local<v8::Value> v8Array);

v8::Local<v8::Value> fannDataToV8Array(FANN::fann_type * data, unsigned int size);

}

