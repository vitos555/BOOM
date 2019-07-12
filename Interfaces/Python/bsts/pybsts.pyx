from model_manager cimport ScalarModelManager, ScalarStateSpaceSpecification, PyBstsOptions, LocalTrend, Matrix, Vector
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref

cdef object matrix_to_nparray(Matrix x):
    return None

cdef Matrix nparray_to_matrix(x):
    cdef Matrix m = Matrix()
    return m

cdef object vector_to_ndarray(Vector x):
    return None

cdef Vector nparray_to_vector(x):
    cdef Vector v = Vector()
    return v

cdef class PyBsts:
    cdef unique_ptr[ScalarModelManager] cpp_scalar_model_manager

    def __cinit__(self, family, xdim, specification, options):
        cdef ScalarModelManager *model = ScalarModelManager.Create(family, xdim)
        cdef ScalarStateSpaceSpecification *cpp_specification = new ScalarStateSpaceSpecification()
        cdef PyBstsOptions *cpp_options = new PyBstsOptions()
        cdef bool status = model.InitializeModel(cpp_specification, cpp_options)
        if status:
            self.cpp_scalar_model_manager.reset(model)
        else:
            raise MemoryError()

    def __init__(self, family, xdim, specification, options):
        pass

    def fit(self, X, y):
        status = deref(self.cpp_scalar_model_manager).fit(nparray_to_matrix(X), nparray_to_vector(y))
        if not status:
            raise Exception()

    def predict(self, X):
        return matrix_to_nparray(deref(self.cpp_scalar_model_manager).predict(nparray_to_matrix(X)))

