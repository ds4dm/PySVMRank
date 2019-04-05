import sys
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, realloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT

from libc.stdio cimport printf
from libc.string cimport strcpy, strlen


def str_sanitize(s):
    return unicode(s).encode('utf-8')

cdef class Model:

    cdef STRUCTMODEL s_model
    cdef STRUCT_LEARN_PARM s_parm
    cdef KERNEL_PARM k_parm
    cdef LEARN_PARM l_parm

    def __init__(self):
        pass

    def __cinit__(self):
        self.s_model.svm_model = NULL

    def __dealloc__(self):
        if self.s_model.svm_model:
            free_model(self.s_model.svm_model, 1)  # release also support vectors

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit(self, xs, ys, groups, params=None):
        cdef SAMPLE sample
        cdef int alg_type

        cdef int i, argc
        cdef char ** argv

        if params is None:
            params = {}

        if ys.ndim == 2:
            ys = np.squeeze(xs, axis=1)
        if groups.ndim == 2:
            groups = np.squeeze(groups, axis=1)

        if xs.ndim != 2:
            raise ValueError(f"2 dimensions expected for argument 'xs' (has {xs.ndim})")
        if ys.ndim != 1:
            raise ValueError(f"1 dimension expected for argument 'ys' (has {ys.ndim})")
        if groups.ndim != 1:
            raise ValueError(f"1 dimension expected for argument 'groups' (has {groups.ndim})")

        xs = xs.astype(np.float32, copy=False)
        ys = ys.astype(np.float32, copy=False)
        groups = groups.astype(np.int32, copy=False)

        # command-line style parameters
        args = [str_sanitize(arg) for k, v in params.items() for arg in [k, v]]
        argc = len(args)
        argv = <char**> my_malloc(sizeof(char*) * argc)
        for i, arg in enumerate(args):
            argv[i] = arg

        # verbosity and struct_verbosity are globally defined
        read_input_parameters(argc, argv, &verbosity, &struct_verbosity,
                              &self.s_parm, &self.l_parm, &self.k_parm,
                              &alg_type)

        free(argv)
        del args  # reference should be kept until now

        if struct_verbosity >= 1:
            print("Loading training examples", flush=True)

        # load (copy) training dataset
        sample = read_struct_examples(xs, ys, groups, &self.s_parm)

        if struct_verbosity >= 1:
            print("Training", flush=True)

        # train
        if alg_type == 0:
            svm_learn_struct(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, NSLACK_ALG)
        elif alg_type == 1:
            svm_learn_struct(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, NSLACK_SHRINK_ALG)
        elif alg_type == 2:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_PRIMAL_ALG)
        elif alg_type == 3:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_DUAL_ALG)
        elif alg_type == 4:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_DUAL_CACHE_ALG)
        elif alg_type == 9:
            svm_learn_struct_joint_custom(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model)
        else:
            exit(1)

        if struct_verbosity >= 1:
            print("Detaching model from training examples (support vectors)", flush=True)

        # copy model, in order to detach support vectors out of sample
        cdef MODEL * tmp = self.s_model.svm_model
        self.s_model.svm_model = copy_model(tmp)
        free_model(tmp, 0)

        if struct_verbosity >= 1:
            print("Releasing training examples", flush=True)

        # release training sample
        free_struct_sample(sample)

        if struct_verbosity >= 1:
            print("Training done", flush=True)

    def predict(self, xs, groups):
        return []

    def write(self, filename="svm_struct_model"):
        if not self.s_model.svm_model:
            raise Exception("There is no model to write.")

        write_struct_model(str_sanitize(filename), &self.s_model, &self.s_parm)


cdef SAMPLE read_struct_examples(
        np.ndarray[np.float32_t, ndim=2] xs,
        np.ndarray[np.float32_t, ndim=1] ys,
        np.ndarray[np.int32_t, ndim=1] groups,
        STRUCT_LEARN_PARM *sparm):
    cdef SAMPLE    sample
    cdef DOC    ** instances
    cdef double  * labels
    cdef int       n, d, i, j

    n = xs.shape[0]
    d = xs.shape[1]

    assert n == ys.shape[0] and n == groups.shape[0]

    # allocate instances and labels
    instances = <DOC**> my_malloc(sizeof(DOC*) * n)
    labels = <double*> my_malloc(sizeof(double) * n)
    for i in range(n):
        labels[i] = ys[i]  # copy from numpy

        # instances should be allocated individually, see create_example()
        instances[i] = <DOC*> my_malloc(sizeof(DOC))
        instances[i].docnum = i
        instances[i].kernelid = i
        instances[i].queryid = groups[i]  # copy from numpy
        instances[i].slackid = 0
        instances[i].costfactor = 0

        # words and vectors should be allocated individually, see create_svector()
        instances[i].fvec = <SVECTOR*> my_malloc(sizeof(SVECTOR))
        instances[i].fvec.twonorm_sq = -1
        instances[i].fvec.userdefined = NULL
        instances[i].fvec.kernel_id = 0
        instances[i].fvec.next = NULL
        instances[i].fvec.factor = 1.0
        instances[i].fvec.dense = NULL
        instances[i].fvec.size = -1
        instances[i].fvec.words = <WORD*> my_malloc(sizeof(WORD) * (d + 1))
        for j in range(d):
            instances[i].fvec.words[j].wnum = j + 1
            instances[i].fvec.words[j].weight = xs[i, j]  # copy from numpy
        instances[i].fvec.words[d].wnum = 0  # end of words flag

    sample = build_sample(labels, instances, n, sparm)

    free(instances)
    free(labels)

    return sample

