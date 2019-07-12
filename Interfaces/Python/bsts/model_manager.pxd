from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "LinAlg/Vector.hpp" namespace "BOOM":
    cdef cppclass Vector:
        Vector() except +

cdef extern from "LinAlg/Vector.hpp" namespace "BOOM":
    cdef cppclass Matrix:
        Matrix() except +

cdef extern from "model_manager.hpp" namespace "BOOM::pybsts":
    cdef cppclass ScalarStateSpaceSpecification:
        ScalarStateSpaceSpecification() except +

    cdef cppclass PyBstsOptions:
        PyBstsOptions() except +
        PyBstsOptions(bool save_state_contributions, bool save_full_state,
            bool save_prediction_errors, int niter, int ping,
            double timeout_threshold_seconds) except +

    cdef cppclass LocalTrend:
        LocalTrend(bool has_intercept, bool has_trend, bool has_slope, 
            bool slope_has_bias, bool student_errors) except +

    cdef cppclass ScalarModelManager:
        ScalarModelManager() except +

        @staticmethod
        ScalarModelManager* Create(string family, int xdim)

        bool InitializeModel(ScalarStateSpaceSpecification *specification, PyBstsOptions *options)
        bool fit(Matrix x, Vector y)
        Matrix predict(Matrix x)


