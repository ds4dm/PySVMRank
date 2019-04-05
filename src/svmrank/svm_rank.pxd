cdef extern from "svm_light/svm_common.h":

    cdef char* VERSION
    cdef char* VERSION_DATE
    cdef int   LINEAR

    ctypedef struct MATRIX:
        pass

    ctypedef struct LEARN_PARM:
        long     type
        double   svm_c
        double   eps
        double   svm_costratio
        double   transduction_posratio
        long     biased_hyperplane
        long     sharedslack
        long     svm_maxqpsize
        long     svm_newvarsinqp
        long     kernel_cache_size
        double   epsilon_crit
        double   epsilon_shrink
        long     svm_iter_to_shrink
        long     maxiter
        long     remove_inconsistent
        long     skip_final_opt_check
        long     compute_loo
        double   rho
        long     xa_depth
        char *   predfile
        char *   alphafile
        double   epsilon_const
        double   epsilon_a
        double   opt_precision
        long     svm_c_steps
        double   svm_c_factor
        double   svm_costratio_unlab
        double   svm_unlabbound
        double * svm_cost
        long     totwords

    ctypedef struct KERNEL_PARM:
        long     kernel_type
        long     poly_degree
        double   rbf_gamma
        double   coef_lin
        double   coef_const
        char *   custom
        MATRIX * gram_matrix
        long     totwords

    ctypedef struct DOC:
        long      docnum
        long      queryid
        double    costfactor
        long      slackid
        long      kernelid
        SVECTOR * fvec

    ctypedef struct SVECTOR:
        WORD    * words
        double    twonorm_sq
        char    * userdefined
        long      kernel_id
        SVECTOR * next
        double    factor
        double  * dense
        long      size

    ctypedef struct WORD:
        int   wnum
        float weight

    ctypedef struct MODEL:
        pass

    long     verbosity
    void     copyright_notice()
    void*    my_malloc(size_t size)
    MODEL*   copy_model(MODEL * m)
    void     free_model(MODEL * m, int deep)


cdef extern from "svm_struct/svm_struct_common.h":

    cdef char* STRUCT_VERSION
    cdef char* STRUCT_VERSION_DATE

    ctypedef struct EXAMPLE:
        pass

    ctypedef struct SAMPLE:
        int n
        EXAMPLE* examples

    long struct_verbosity


cdef extern from "svm_struct/svm_struct_learn.h":

    cdef int MARGIN_RESCALING
    cdef int SLACK_RESCALING

    cdef int NSLACK_ALG
    cdef int NSLACK_SHRINK_ALG
    cdef int ONESLACK_PRIMAL_ALG
    cdef int ONESLACK_DUAL_ALG
    cdef int ONESLACK_DUAL_CACHE_ALG

    void svm_learn_struct(
            SAMPLE sample, STRUCT_LEARN_PARM *sparm,
            LEARN_PARM *lparm, KERNEL_PARM *kparm, 
            STRUCTMODEL *sm, int alg_type)
    void svm_learn_struct_joint(
            SAMPLE sample, STRUCT_LEARN_PARM *sparm,
            LEARN_PARM *lparm, KERNEL_PARM *kparm, 
            STRUCTMODEL *sm, int alg_type)
    void svm_learn_struct_joint_custom(
            SAMPLE sample, STRUCT_LEARN_PARM *sparm,
            LEARN_PARM *lparm, KERNEL_PARM *kparm, 
            STRUCTMODEL *sm)


cdef extern from "svm_struct_api_types.h":

    cdef char* INST_NAME
    cdef char* INST_VERSION
    cdef char* INST_VERSION_DATE
    cdef float DEFAULT_EPS
    cdef int   DEFAULT_ALG_TYPE
    cdef int   DEFAULT_LOSS_FCT
    cdef int   DEFAULT_RESCALING

    ctypedef struct STRUCT_LEARN_PARM:
        double  epsilon
        double  newconstretrain
        int     ccache_size
        double  batch_size
        double  C
        char ** custom_argv
        int     custom_argc
        int     slack_norm
        int     loss_type
        int     loss_function
        int     num_features

    ctypedef struct STRUCTMODEL:
        MODEL * svm_model


cdef extern from "svm_struct_api.h":

    void print_struct_help()
    void parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
    void free_struct_model(STRUCTMODEL sm)
    void free_struct_sample(SAMPLE s)
    void  write_struct_model(
            char *file, STRUCTMODEL *sm, 
			STRUCT_LEARN_PARM *sparm);

cdef extern from "utilities.h":

    void read_input_parameters(
            int argc,char *argv[],
            long *verbosity,long *struct_verbosity,
            STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
            KERNEL_PARM *kernel_parm, int *alg_type)
    SAMPLE build_sample(
            double * labels, DOC ** instances, int n,
            STRUCT_LEARN_PARM * sparm)
