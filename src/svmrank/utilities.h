
#ifndef utilities_h
#define utilities_h

#include "svm_struct_api_types.h"
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"

int read_input_parameters(
        int argc, char *argv[],
        long *verbosity, long *struct_verbosity,
        STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
        KERNEL_PARM *kernel_parm, int *alg_type);

SAMPLE build_sample(
        double * labels, DOC ** instances, int n,
        STRUCT_LEARN_PARM * sparm);

void print_help();
void * my_calloc(size_t num_elements, size_t element_size);
void * my_realloc(void * ptr, size_t size);

#endif
