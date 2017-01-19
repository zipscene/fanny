#include <nan.h>
#include "fanny.h"

NAN_MODULE_INIT(init) {
	fanny::FANN::Init(target);
}

NODE_MODULE(fanny, init);

