import sys
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT

from libc.stdio cimport printf
from libc.string cimport strcpy, strlen


cdef class Model:

    cdef STRUCTMODEL structmodel

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit(self, x, y, params=None):
        cdef SAMPLE sample
        cdef LEARN_PARM learn_parm
        cdef KERNEL_PARM kernel_parm
        cdef STRUCT_LEARN_PARM struct_parm

        # cdef int argc
        # cdef char ** argv
        # cdef char trainfile[200]
        # cdef char modelfile[200]
        cdef int i

        # # c-style command-line args
        # args = ['']
        # for name, value in params.items():
        #     args.append(name)
        #     args.append(value)

        # argc = len(args)
        # argv = <char**> malloc(sizeof(char*) * argc)
        # for i, arg in enumerate(args):
        #     arg = unicode(arg).encode('utf-8')
        #     argv[i] = arg
        #     Py_INCREF(arg)

        cdef int alg_type

        if params is None:
            params = {}

        # verbosity and struct_verbosity are globally defined
        read_input_parameters(
            params, &verbosity,
            # argc, argv, trainfile, modelfile, &verbosity,
            &struct_verbosity, &struct_parm, &learn_parm,
            &kernel_parm, &alg_type)

    #   /* read the training examples */
    #   sample=read_struct_examples(trainfile,&struct_parm)
    #   
    #   /* Do the learning and return structmodel. */
    #   if(alg_type == 0)
    #     svm_learn_struct(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,NSLACK_ALG)
    #   else if(alg_type == 1)
    #     svm_learn_struct(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,NSLACK_SHRINK_ALG)
    #   else if(alg_type == 2)
    #     svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_PRIMAL_ALG)
    #   else if(alg_type == 3)
    #     svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_DUAL_ALG)
    #   else if(alg_type == 4)
    #     svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_DUAL_CACHE_ALG)
    #   else if(alg_type == 9)
    #     svm_learn_struct_joint_custom(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel)
    #   else
    #     exit(1)
    # 
    #   /* Warning: The model contains references to the original data 'docs'.
    #      If you want to free the original data, and only keep the model, you 
    #      have to make a deep copy of 'model'. */
    #   write_struct_model(modelfile,&structmodel,&struct_parm)
    # 
    #   free_struct_sample(sample)
    #   free_struct_model(structmodel)

        # for i in range(argc):
        #     Py_DECREF(argv[i])

        # free(argv)

    def predict(self, x):
        return []


cdef void read_input_parameters(
        dict params,
        long *verbosity, long *struct_verbosity, 
        STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
        KERNEL_PARM *kernel_parm, int *alg_type):
    cdef long i
    cdef char type[100]

    verbosity[0] = 0  # verbosity for svm_light
    struct_verbosity[0] = 1  # verbosity for struct learning portion

    alg_type[0] = DEFAULT_ALG_TYPE
    strcpy(type, "c")

    struct_parm.C = -0.01
    struct_parm.slack_norm = 1
    struct_parm.epsilon = DEFAULT_EPS
    struct_parm.custom_argc = 0
    struct_parm.loss_function = DEFAULT_LOSS_FCT
    struct_parm.loss_type = DEFAULT_RESCALING
    struct_parm.newconstretrain = 100
    struct_parm.ccache_size = 5
    struct_parm.batch_size = 100

    strcpy(learn_parm.predfile, "trans_predictions")
    strcpy(learn_parm.alphafile, "")
    learn_parm.biased_hyperplane = 1
    learn_parm.remove_inconsistent = 0
    learn_parm.skip_final_opt_check = 0
    learn_parm.svm_maxqpsize = 10
    learn_parm.svm_newvarsinqp = 0
    learn_parm.svm_iter_to_shrink = -9999
    learn_parm.maxiter = 100000
    learn_parm.kernel_cache_size = 40
    learn_parm.svm_c = 99999999  # overridden by struct_parm.C
    learn_parm.eps = 0.001  # overridden by struct_parm.epsilon
    learn_parm.transduction_posratio = -1.0
    learn_parm.svm_costratio = 1.0
    learn_parm.svm_costratio_unlab = 1.0
    learn_parm.svm_unlabbound = 1E-5
    learn_parm.epsilon_crit = 0.001
    learn_parm.epsilon_a = 1E-10  # changed from 1e-15
    learn_parm.compute_loo = 0
    learn_parm.rho = 1.0
    learn_parm.xa_depth = 0

    kernel_parm.kernel_type = 0
    kernel_parm.poly_degree = 3
    kernel_parm.rbf_gamma = 1.0
    kernel_parm.coef_lin = 1
    kernel_parm.coef_const = 1
    strcpy(kernel_parm.custom, "empty")

    for key, value in params.items():
        if key == '-?':
            print_help()
            exit(0)
        elif key == '-a':
            strcpy(learn_parm.alphafile, str(value))
        elif key == '-c':
            struct_parm.C = float(value)
        elif key == '-p':
            struct_parm.slack_norm = long(value)
        elif key == '-e':
            struct_parm.epsilon = float(value)
        elif key == '-k':
            struct_parm.newconstretrain = long(value)
        elif key == '-h':
            learn_parm.svm_iter_to_shrink = long(value)
        elif key == '-#':
            learn_parm.maxiter = long(value)
        elif key == '-m':
            learn_parm.kernel_cache_size = long(value)
        elif key == '-w':
            alg_type[0] = long(value)
        elif key == '-o':
            struct_parm.loss_type = long(value)
        elif key == '-n':
            learn_parm.svm_newvarsinqp = long(value)
        elif key == '-q':
            learn_parm.svm_maxqpsize = long(value)
        elif key == '-l':
            struct_parm.loss_function = long(value)
        elif key == '-f':
            struct_parm.ccache_size = long(value)
        elif key == '-b':
            struct_parm.batch_size = float(value)
        elif key == '-t':
            kernel_parm.kernel_type = long(value)
        elif key == '-d':
            kernel_parm.poly_degree = long(value)
        elif key == '-g':
            kernel_parm.rbf_gamma = float(value)
        elif key == '-s':
            kernel_parm.coef_lin = float(value)
        elif key == '-r':
            kernel_parm.coef_const = float(value)
        elif key == '-u':
            strcpy(kernel_parm.custom, str(value))
        elif key == '-v':
            struct_verbosity[0] = long(value)
        elif key == '-y':
            verbosity[0] = long(value)
        elif key[:2] == '--':
            strcpy(struct_parm.custom_argv[struct_parm.custom_argc], str(key))
            struct_parm.custom_argc += 1
            strcpy(struct_parm.custom_argv[struct_parm.custom_argc], str(value))
            struct_parm.custom_argc += 1
        else:
            print()
            print(f"Unrecognized option: '{key}'")
            print()
            print_help()
            exit(0)

    if learn_parm.svm_iter_to_shrink == -9999:
        learn_parm.svm_iter_to_shrink = 100

    if learn_parm.skip_final_opt_check and kernel_parm.kernel_type == LINEAR:
        print()
        print("It does not make sense to skip the final optimality check for linear kernels.")
        print()
        learn_parm.skip_final_opt_check = 0

    if learn_parm.skip_final_opt_check and learn_parm.remove_inconsistent:
        print()
        print("It is necessary to do the final optimality check when removing inconsistent examples.")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.svm_maxqpsize < 2:
        print()
        print(f"Maximum size of QP-subproblems not in valid range: {learn_parm.svm_maxqpsize} [2..]")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.svm_maxqpsize < learn_parm.svm_newvarsinqp:
        print()
        print(f"Maximum size of QP-subproblems [{learn_parm.svm_maxqpsize}] must be larger than the number of new variables [{learn_parm.svm_newvarsinqp}] entering the working set in each iteration.")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.svm_iter_to_shrink < 1:
        print()
        print(f"Maximum number of iterations for shrinking not in valid range: {learn_parm.svm_iter_to_shrink} [1,..]")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.C < 0:
        print()
        print(f"You have to specify a value for the parameter '-c' (C>0)!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if alg_type[0] not in (0, 1, 2, 3, 4, 9):
        print()
        print(f"Algorithm type must be either '0', '1', '2', '3', '4', or '9'!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.transduction_posratio > 1:
        print()
        print("The fraction of unlabeled examples to classify as positives must")
        print("be less than 1.0 !!!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.svm_costratio <= 0:
        print()
        print("The COSTRATIO parameter must be greater than zero!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.epsilon <= 0:
        print()
        print("The epsilon parameter must be greater than zero!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.ccache_size <= 0 and alg_type[0] == 4:
        print()
        print("The cache size must be at least 1!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.batch_size <= 0 or struct_parm.batch_size > 100 and alg_type[0] == 4:
        print()
        print("The batch size must be in the interval ]0,100]!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.slack_norm not in (1, 2):
        print()
        print("The norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if struct_parm.loss_type not in (SLACK_RESCALING, MARGIN_RESCALING):
        print()
        print(f"The loss type must be either {SLACK_RESCALING} (slack rescaling) or {MARGIN_RESCALING} (margin rescaling)!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.rho < 0:
        print()
        print("The parameter rho for xi/alpha-estimates and leave-one-out pruning must be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)!")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    if learn_parm.xa_depth < 0 or learn_parm.xa_depth > 100:
        print()
        print("The parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero for switching to the conventional xa/estimates described in T. Joachims, Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)")
        print()
        input("(enter to continue)")
        print_help()
        exit(0)

    parse_struct_parameters(struct_parm)


cdef void print_help():
    print()
    print(f"SVM-struct learning module: {str(INST_NAME)}, {str(INST_VERSION)}, {str(INST_VERSION_DATE)}")
    print(f"   includes SVM-struct {str(STRUCT_VERSION)} for learning complex outputs, {str(STRUCT_VERSION_DATE)}")
    print(f"   includes SVM-light {str(VERSION)} quadratic optimizer, {str(VERSION_DATE)}")
    copyright_notice()
    print(f"   usage: svm_struct_learn [options] example_file model_file")
    print(f"Arguments:")
    print(f"         example_file-> file with training data")
    print(f"         model_file  -> file to store learned decision rule in")

    print(f"General Options:")
    print(f"         -?          -> this help")
    print(f"         -v [0..3]   -> verbosity level (default 1)")
    print(f"         -y [0..3]   -> verbosity level for svm_light (default 0)")
    print(f"Learning Options:")
    print(f"         -c float    -> C: trade-off between training error")
    print(f"                        and margin (default 0.01)")
    print(f"         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,")
    print(f"                        use 2 for squared slacks. (default 1)")
    print(f"         -o [1,2]    -> Rescaling method to use for loss.")
    print(f"                        1: slack rescaling")
    print(f"                        2: margin rescaling")
    print(f"                        (default {DEFAULT_RESCALING})")
    print(f"         -l [0..]    -> Loss function to use.")
    print(f"                        0: zero/one loss")
    print(f"                        ?: see below in application specific options")
    print(f"                        (default {DEFAULT_LOSS_FCT})")
    print(f"Optimization Options (see [2][5]):")
    print(f"         -w [0,..,9] -> choice of structural learning algorithm (default {DEFAULT_ALG_TYPE}):")
    print(f"                        0: n-slack algorithm described in [2]")
    print(f"                        1: n-slack algorithm with shrinking heuristic")
    print(f"                        2: 1-slack algorithm (primal) described in [5]")
    print(f"                        3: 1-slack algorithm (dual) described in [5]")
    print(f"                        4: 1-slack algorithm (dual) with constraint cache [5]")
    print(f"                        9: custom algorithm in svm_struct_learn_custom.c")
    print(f"         -e float    -> epsilon: allow that tolerance for termination")
    print(f"                        criterion (default {DEFAULT_EPS})")
    print(f"         -k [1..]    -> number of new constraints to accumulate before")
    print(f"                        recomputing the QP solution (default 100)")
    print(f"                        (-w 0 and 1 only)")
    print(f"         -f [5..]    -> number of constraints to cache for each example")
    print(f"                        (default 5) (used with -w 4)")
    print(f"         -b [1..100] -> percentage of training set for which to refresh cache")
    print(f"                        when no epsilon violated constraint can be constructed")
    print(f"                        from current cache (default 100%%) (used with -w 4)")
    print(f"SVM-light Options for Solving QP Subproblems (see [3]):")
    print(f"         -n [2..q]   -> number of new variables entering the working set")
    print(f"                        in each svm-light iteration (default n = q).")
    print(f"                        Set n < q to prevent zig-zagging.")
    print(f"         -m [5..]    -> size of svm-light cache for kernel evaluations in MB")
    print(f"                        (default 40) (used only for -w 1 with kernels)")
    print(f"         -h [5..]    -> number of svm-light iterations a variable needs to be")
    print(f"                        optimal before considered for shrinking (default 100)")
    print(f"         -# int      -> terminate svm-light QP subproblem optimization, if no")
    print(f"                        progress after this number of iterations.")
    print(f"                        (default 100000)")
    print(f"Kernel Options:")
    print(f"         -t int      -> type of kernel function:")
    print(f"                        0: linear (default)")
    print(f"                        1: polynomial (s a*b+c)^d")
    print(f"                        2: radial basis function exp(-gamma ||a-b||^2)")
    print(f"                        3: sigmoid tanh(s a*b + c)")
    print(f"                        4: user defined kernel from kernel.h")
    print(f"         -d int      -> parameter d in polynomial kernel")
    print(f"         -g float    -> parameter gamma in rbf kernel")
    print(f"         -s float    -> parameter s in sigmoid/poly kernel")
    print(f"         -r float    -> parameter c in sigmoid/poly kernel")
    print(f"         -u string   -> parameter of user defined kernel")
    print(f"Output Options:")
    print(f"         -a string   -> write all alphas to this file after learning")
    print(f"                        (in the same order as in the training set)")
    print(f"Application-Specific Options:")
    print_struct_help()
    input("(enter to continue)")

    print()
    print(f"More details in:")
    print(f"[1] T. Joachims, Learning to Align Sequences: A Maximum Margin Aproach.")
    print(f"    Technical Report, September, 2003.")
    print(f"[2] I. Tsochantaridis, T. Joachims, T. Hofmann, and Y. Altun, Large Margin")
    print(f"    Methods for Structured and Interdependent Output Variables, Journal")
    print(f"    of Machine Learning Research (JMLR), Vol. 6(Sep):1453-1484, 2005.")
    print(f"[3] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in")
    print(f"    Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and")
    print(f"    A. Smola (ed.), MIT Press, 1999.")
    print(f"[4] T. Joachims, Learning to Classify Text Using Support Vector")
    print(f"    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,")
    print(f"    2002.")
    print(f"[5] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural")
    print(f"    SVMs, Machine Learning Journal, to appear.")

