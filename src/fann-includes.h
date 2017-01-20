#if defined FANNY_FLOAT
#include <floatfann.h>
#elif defined FANNY_DOUBLE
#include <doublefann.h>
#elif defined FANNY_FIXED
#include <fixedfann.h>
#else
#error "Must define one of FANNY_FLOAT, FANNY_DOUBLE, FANNY_FIXED"
#include <floatfann.h>
#endif

#include <fann_cpp.h>

