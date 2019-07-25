from model_manager cimport ScalarModelManager, ScalarManagedModel, PythonListIoManager, \
    ScalarStateSpaceSpecification, ModelOptions, LocalTrend, Matrix, Vector
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cpython.version cimport PY_MAJOR_VERSION

import numpy as np
cimport numpy as np

np.import_array()

cdef bytes _bytes(s):
    if type(s) is unicode:
        return <bytes>((<unicode>s).encode("ascii", errors="ignore"))

    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        return s

    elif isinstance(s, unicode):
        return <bytes>unicode(s).encode("ascii", errors="ignore")

    elif isinstance(s, str):
        return <bytes>s

    else:
        raise TypeError("Could not convert to unicode.")

cdef np.ndarray[double, ndim=2] matrix_to_nparray(Matrix x) except +:
    cdef double[:, :] cy_array = np.empty((x.nrow(), x.ncol()), dtype=np.float64)
    cdef double[:, :] tmp = <double[:x.nrow(), :x.ncol()]>(x.data())
    cy_array[:, ::1] = tmp.copy()
    return np.asarray(cy_array)

cdef Matrix nparray_to_matrix(np.ndarray[double, ndim=2] x) except +:
    cdef Matrix ret = Matrix(x.shape[0], x.shape[1], <const double *>x.data)
    return ret

cdef np.ndarray[double, ndim=1] vector_to_ndarray(Vector x) except +:
    cdef double[:] cy_vector = <double[:x.size()]>(x.data())
    cdef double[:] ret = cy_vector.copy()
    return np.asarray(ret)


cdef Vector nparray_to_vector(np.ndarray[double, ndim=1] x) except +:
    cdef Vector v = Vector(<vector[double]>x)
    return v

cdef class PyBsts:
    cdef unique_ptr[ScalarModelManager] cpp_scalar_model_manager
    cdef unique_ptr[ScalarManagedModel] cpp_model
    cdef shared_ptr[PythonListIoManager] cpp_io_manager

    def __cinit__(self, family, xdim, specification, options):
        self.cpp_scalar_model_manager.reset(ScalarModelManager.Create(_bytes(family), xdim))
        self.cpp_io_manager.reset(new PythonListIoManager())
        self.cpp_model.reset(deref(self.cpp_scalar_model_manager).CreateModel(
            new ScalarStateSpaceSpecification(),
            new ModelOptions(),
            self.cpp_io_manager))

    def crepr(self):
        return <bytes>deref(self.cpp_io_manager).repr().c_str()

    def __repr__(self):
        return self.crepr().decode("UTF-8")

    def fit(self, X, y):
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("BSTS.fit can not be called with empty data.")
        status = deref(self.cpp_model).fit(nparray_to_vector(y), nparray_to_matrix(X))
        if not status:
            raise Exception("Failure during model fit.")

    def predict(self, X):
        if X is None or len(X) == 0:
            ValueError("BSTS.predict can not be called with empty data.")
        return matrix_to_nparray(deref(self.cpp_model).predict(nparray_to_matrix(X)))

