#include <nan.h>
#include "fanny.h"
#include "training-data.h"

NAN_MODULE_INIT(init) {
	fanny::FANNY::Init(target);
	fanny::TrainingData::Init(target);
}

NODE_MODULE(fanny, init);

