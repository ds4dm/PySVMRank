
#ifndef utilities_h
#define utilities_h

#include "svm_struct_api_types.h"
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"

void read_input_parameters(
        int argc, char *argv[],
        long *verbosity, long *struct_verbosity,
        STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
        KERNEL_PARM *kernel_parm, int *alg_type);

SAMPLE build_sample(
        double * labels, DOC ** instances, int n,
        STRUCT_LEARN_PARM * sparm);

void wait_any_key();

void print_help();

#endif
