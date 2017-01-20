#include <nan.h>
#include "fanny.h"

NAN_MODULE_INIT(init) {
	fanny::FANNY::Init(target);
}

NODE_MODULE(fanny, init);

