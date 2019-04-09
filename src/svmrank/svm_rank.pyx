import sys
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, realloc, calloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT

from libc.stdio cimport printf
from libc.string cimport strcpy, strlen


def str_sanitize(s):
    return unicode(s).encode('utf-8')

def help():
    print_help()

cdef class Model:

    cdef STRUCTMODEL s_model
    cdef STRUCT_LEARN_PARM s_parm
    cdef KERNEL_PARM k_parm
    cdef LEARN_PARM l_parm
    cdef int alg_type
    cdef dict params

    def __init__(self, params=None):
        self.params = {}
        if params is not None:
           self.set_params(params) 

    def __cinit__(self):
        self.s_model.svm_model = NULL

    def __dealloc__(self):
        if self.s_model.svm_model:
            free_model(self.s_model.svm_model, 1)  # release also support vectors

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def set_params(self, params):
        for k, v in params.items():
            self.params[k] = v

    def _apply_params(self):
        cdef int i, argc
        cdef char ** argv

        # command-line style parameters
        args = [str_sanitize(arg) for k, v in self.params.items() for arg in [k, v]]
        argc = len(args)
        argv = <char**> malloc(sizeof(char*) * argc)
        if argv is NULL:
            raise MemoryError

        for i, arg in enumerate(args):
            argv[i] = arg  
            Py_INCREF(arg)

        # verbosity and struct_verbosity are globally defined...
        error = not read_input_parameters(
            argc, argv, &verbosity, &struct_verbosity,
            &self.s_parm, &self.l_parm, &self.k_parm, &self.alg_type)

        free(argv)
        for arg in args:
            Py_DECREF(arg)

        if error:
            raise ValueError("Illegal parameters. See svmrank.help() for a list of available parameters.")

    def fit(self, xs, ys, groups):
        cdef SAMPLE sample

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

        assert xs.shape[0] == groups.shape[0] and xs.shape[0] == ys.shape[0]

        self._apply_params()
        parse_struct_parameters(&self.s_parm)

        xs = xs.astype(np.float32, copy=False)
        ys = ys.astype(np.float32, copy=False)
        groups = groups.astype(np.int32, copy=False)

        # load (copy) training dataset
        if struct_verbosity >= 1:
            print("Loading training examples", flush=True)

        sample = read_struct_examples(xs, ys, groups, &self.s_parm)

        # train
        if struct_verbosity >= 1:
            print("Training", flush=True)

        if self.alg_type == 0:
            svm_learn_struct(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, NSLACK_ALG)
        elif self.alg_type == 1:
            svm_learn_struct(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, NSLACK_SHRINK_ALG)
        elif self.alg_type == 2:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_PRIMAL_ALG)
        elif self.alg_type == 3:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_DUAL_ALG)
        elif self.alg_type == 4:
            svm_learn_struct_joint(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model, ONESLACK_DUAL_CACHE_ALG)
        elif self.alg_type == 9:
            svm_learn_struct_joint_custom(sample, &self.s_parm, &self.l_parm, &self.k_parm, &self.s_model)
        else:
            raise ValueError("Incorrect algorithm type: '{self.alg_type}'")

        # copy model, in order to detach support vectors out of sample
        if struct_verbosity >= 1:
            print("Detaching model from training examples (support vectors)", flush=True)

        cdef MODEL * tmp = self.s_model.svm_model
        self.s_model.svm_model = copy_model(tmp)
        free_model(tmp, 0)

        # release training sample
        if struct_verbosity >= 1:
            print("Releasing training examples", flush=True)

        free_struct_sample(sample)

        if struct_verbosity >= 1:
            print("Training done", flush=True)

    def predict(self, xs, groups):
        cdef SAMPLE sample
        cdef int i, j, k
        cdef LABEL y

        if self.s_model.svm_model is NULL:
            raise ValueError("There is no model to use for prediction.")

        if groups.ndim == 2:
            groups = np.squeeze(groups, axis=1)

        if xs.ndim != 2:
            raise ValueError(f"2 dimensions expected for argument 'xs' (has {xs.ndim})")
        if groups.ndim != 1:
            raise ValueError(f"1 dimension expected for argument 'groups' (has {groups.ndim})")

        assert xs.shape[0] == groups.shape[0]

        self._apply_params()
        parse_struct_parameters_classify(&self.s_parm)

        xs = xs.astype(np.float32, copy=False)
        groups = groups.astype(np.int32, copy=False)

        if struct_verbosity >= 1:
            print("Loading test examples...", flush=True)

        sample = read_struct_examples(xs, None, groups, &self.s_parm)

        if struct_verbosity >= 1:
            print("Ranking test examples...")

        preds = np.empty(groups.shape, np.float32)
        k = 0
        for i in range(sample.n):
            y = classify_struct_example(sample.examples[i].x, &self.s_model, &self.s_parm)
            for j in range(y.totdoc):
                preds[k] = y._class[j]
            free_label(y)

        free_struct_sample(sample)

        return preds

    def loss(self, ys, preds):
        #   l=loss(testsample.examples[i].y,y,&sparm);
        raise NotImplementedError

    def read(self, filename="svm_struct_model"):
        if struct_verbosity >= 1:
            print("Reading model", flush=True)
        self.s_model = read_struct_model(str_sanitize(filename), &self.s_parm)

        if self.s_model.svm_model.kernel_parm.kernel_type == LINEAR:
          add_weight_vector_to_linear_model(self.s_model.svm_model)
          self.s_model.w = self.s_model.svm_model.lin_weights

    def write(self, filename="svm_struct_model"):
        if self.s_model.svm_model is NULL:
            raise ValueError("There is no model to write.")

        if struct_verbosity >= 1:
            print("Writing model", flush=True)
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

    # allocate instances and labels
    instances = <DOC**> malloc(sizeof(DOC*) * n)
    if instances is NULL:
        raise MemoryError

    labels = <double*> calloc(n, sizeof(double))  # zero initialized
    if labels is NULL:
        raise MemoryError

    for i in range(n):
        if ys is not None:
            labels[i] = ys[i]  # copy from numpy

        # instances should be allocated individually, see create_example()
        instances[i] = <DOC*> malloc(sizeof(DOC))
        if instances[i] is NULL:
            raise MemoryError

        instances[i].docnum = i
        instances[i].kernelid = i
        instances[i].queryid = groups[i]  # copy from numpy
        instances[i].slackid = 0
        instances[i].costfactor = 0

        # words and vectors should be allocated individually, see create_svector()
        instances[i].fvec = <SVECTOR*> malloc(sizeof(SVECTOR))
        if instances[i].fvec is NULL:
            raise MemoryError

        instances[i].fvec.twonorm_sq = -1
        instances[i].fvec.userdefined = NULL
        instances[i].fvec.kernel_id = 0
        instances[i].fvec.next = NULL
        instances[i].fvec.factor = 1.0
        instances[i].fvec.dense = NULL
        instances[i].fvec.size = -1
        instances[i].fvec.words = <WORD*> malloc(sizeof(WORD) * (d + 1))
        if instances[i].fvec.words is NULL:
            raise MemoryError

        for j in range(d):
            instances[i].fvec.words[j].wnum = j + 1
            instances[i].fvec.words[j].weight = xs[i, j]  # copy from numpy
        instances[i].fvec.words[d].wnum = 0  # end of words flag

    sample = build_sample(labels, instances, n, sparm)

    free(instances)
    free(labels)

    return sample

