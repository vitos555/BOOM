from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

cdef extern from "LinAlg/Vector.hpp" namespace "BOOM":
    cdef cppclass Vector:
        Vector() except +
        Vector(const vector[double] data) except +
        int size()
        double * data()

cdef extern from "LinAlg/Vector.hpp" namespace "BOOM":
    cdef cppclass Matrix:
        Matrix() except +
        Matrix(int ncols, int nrows, const double *data) except +
        int ncol()
        int nrow()
        double * data()

cdef extern from "Models/StateSpace/StateSpaceModelBase.hpp" namespace "BOOM":
    cdef cppclass ScalarStateSpaceModelBase

cdef extern from "list_io.hpp" namespace "BOOM":
    cdef cppclass PythonListIoManager:
        PythonListIoManager() except +
        string repr()


cdef extern from "model_manager.hpp" namespace "BOOM::pybsts":
    cdef cppclass ScalarManagedModel:
        bool fit(Vector y, Matrix x)
        Matrix predict(Matrix x)

    cdef cppclass ScalarStateSpaceSpecification:
        ScalarStateSpaceSpecification() except +

    cdef cppclass ModelOptions:
        ModelOptions() except +
        ModelOptions(bool save_state_contributions, bool save_full_state,
            bool save_prediction_errors, int niter, int ping,
            double timeout_threshold_seconds) except +

    cdef cppclass LocalTrend:
        LocalTrend(bool has_intercept, bool has_trend, bool has_slope, 
            bool slope_has_bias, bool student_errors) except +


    cdef cppclass ScalarModelManager:
        @staticmethod
        ScalarModelManager* Create(string family, int xdim)

        ScalarManagedModel* CreateModel(const ScalarStateSpaceSpecification *specification, 
            ModelOptions *options, 
            shared_ptr[PythonListIoManager] io_manager)


