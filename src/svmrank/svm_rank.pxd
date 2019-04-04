cdef extern from "svm_light/svm_common.h":

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

    long verbosity
    void copyright_notice()

    cdef char* VERSION
    cdef char* VERSION_DATE
    cdef int   LINEAR


cdef extern from "svm_struct/svm_struct_common.h":

    ctypedef struct EXAMPLE:
        pass

    ctypedef struct SAMPLE:
        int n
        EXAMPLE* examples

    long struct_verbosity

    cdef char* STRUCT_VERSION
    cdef char* STRUCT_VERSION_DATE


cdef extern from "svm_struct/svm_struct_learn.h":

    cdef int MARGIN_RESCALING
    cdef int SLACK_RESCALING


cdef extern from "svm_struct_api_types.h":

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
        pass

    cdef char* INST_NAME
    cdef char* INST_VERSION
    cdef char* INST_VERSION_DATE
    cdef float DEFAULT_EPS
    cdef int   DEFAULT_ALG_TYPE
    cdef int   DEFAULT_LOSS_FCT
    cdef int   DEFAULT_RESCALING


cdef extern from "svm_struct_api.h":

    void print_struct_help()
    void parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
