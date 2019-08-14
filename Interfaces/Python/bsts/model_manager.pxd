from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp cimport bool

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[PriorSpecification] move(unique_ptr[PriorSpecification])
    cdef shared_ptr[PriorSpecification] move(shared_ptr[PriorSpecification])
    cdef unique_ptr[LocalTrendSpecification] move(unique_ptr[LocalTrendSpecification])
    cdef shared_ptr[LocalTrendSpecification] move(shared_ptr[LocalTrendSpecification])
    cdef unique_ptr[HierarchicalModelSpecification] move(unique_ptr[HierarchicalModelSpecification])
    cdef shared_ptr[HierarchicalModelSpecification] move(shared_ptr[HierarchicalModelSpecification])
    cdef unique_ptr[OdaOptions] move(unique_ptr[OdaOptions])
    cdef shared_ptr[OdaOptions] move(shared_ptr[OdaOptions])
    cdef unique_ptr[DoublePrior] move(unique_ptr[DoublePrior])
    cdef shared_ptr[DoublePrior] move(shared_ptr[DoublePrior])
    cdef unique_ptr[ModelOptions] move(unique_ptr[ModelOptions])
    cdef shared_ptr[ModelOptions] move(shared_ptr[ModelOptions])

cdef extern from "LinAlg/Vector.hpp" namespace "BOOM":
    cdef cppclass Vector:
        Vector() except +
        Vector(const vector[double] data) except +
        Vector(double val, int size) except +
        int size()
        double * data()


cdef extern from "LinAlg/Matrix.hpp" namespace "BOOM":
    cdef cppclass Matrix:
        Matrix() except +
        Matrix(int ncols, int nrows, const double *data) except +
        Matrix(int ncols, int nrows, const double *data, bool by_row) except +
        int ncol()
        int nrow()
        double * data()

cdef extern from "LinAlg/SpdMatrix.hpp" namespace "BOOM":
    cdef cppclass SpdMatrix:
        SpdMatrix() except +
        SpdMatrix(const Matrix x, bool check) except +
        int dim()
        double * data()

cdef extern from "cpputil/math_utils.hpp" namespace "BOOM":
    cdef double infinity()

cdef extern from "Models/StateSpace/StateSpaceModelBase.hpp" namespace "BOOM":
    cdef cppclass ScalarStateSpaceModelBase


cdef extern from "list_io.hpp" namespace "BOOM":
    cdef cppclass PythonListIoManager:
        PythonListIoManager() except +
        string repr() except +
        Vector results(string name) except +


cdef extern from "prior_specification.hpp" namespace "BOOM::pybsts":
    cdef cppclass DoublePrior:
        DoublePrior(const string family, double a, double b, double a_truncation, double b_truncation) except +

    cdef cppclass PriorSpecification:
        PriorSpecification() except +
        PriorSpecification(
            Vector prior_inclusion_probabilities,
            Vector prior_mean,
            SpdMatrix prior_precision,
            Vector prior_variance_diagonal,
            int max_flips,
            double initial_value,
            double mu,
            double prior_df,
            double prior_guess,
            double sigma_guess,
            double sigma_upper_limit,
            bool truncate,
            bool positive,
            bool fixed) except +


cdef extern from "model_manager.hpp" namespace "BOOM::pybsts":
    void seed_global_rng(int seed) except +

    cdef cppclass ScalarManagedModel:
        void seed_internal_rng(int seed) except +
        void seed_internal_rng() except +
        bool fit(Vector y, Matrix x) except +
        bool fit(Vector y, Matrix x, vector[bool] response_is_observed) except +
        bool fit(Vector y, Matrix x, vector[bool] response_is_observed, vector[int] timestamp_indices) except +
        Matrix predict(Matrix x) except +
        Matrix predict(Matrix x, vector[int] indices) except +

    cdef cppclass ScalarStateSpaceSpecification:
        ScalarStateSpaceSpecification(
            unique_ptr[PriorSpecification] initial_state_prior,
            unique_ptr[PriorSpecification] sigma_prior,
            unique_ptr[PriorSpecification] seasonal_sigma_prior,
            unique_ptr[PriorSpecification] predictors_prior,
            unique_ptr[LocalTrendSpecification] local_trend,
            const string bma_method, unique_ptr[OdaOptions] oda_options,
            vector[SeasonSpecification] seasons,
            unique_ptr[HierarchicalModelSpecification] hierarchical_regression_specification,
            unique_ptr[PriorSpecification] ar_prior,
            const vector[string] predictor_names,
            int ar_order, bool dynamic_regression) except +

    cdef cppclass ModelOptions:
        ModelOptions() except +
        ModelOptions(bool save_state_contributions, bool save_full_state,
              bool save_prediction_errors, int niter, int ping,
              int burn, int forecast_horizon,
              double timeout_threshold_seconds) except +

    cdef cppclass LocalTrendSpecification:
        LocalTrendSpecification(
            unique_ptr[PriorSpecification] trend_prior,
            unique_ptr[PriorSpecification] slope_prior,
            unique_ptr[PriorSpecification] slope_bias_prior,
            unique_ptr[DoublePrior] trend_df_prior,
            unique_ptr[DoublePrior] slope_df_prior,
            unique_ptr[PriorSpecification] slope_ar1_prior,
            bool static_intercept, bool student_errors) except +

    cdef cppclass HierarchicalModelSpecification:
        HierarchicalModelSpecification(
            unique_ptr[DoublePrior] sigma_mean_prior,
            unique_ptr[DoublePrior] shrinkage_prior) except +

    cdef cppclass SeasonSpecification:
        SeasonSpecification(int number_of_seasons, int duration) except +

    cdef cppclass OdaOptions:
        OdaOptions(double eigenvalue_fudge_factor, double fallback_probability) except +

    cdef cppclass ScalarModelManager:
        @staticmethod
        ScalarModelManager* Create(string family, int xdim) except +

        ScalarManagedModel* CreateModel(const ScalarStateSpaceSpecification *specification, 
            ModelOptions *options, 
            shared_ptr[PythonListIoManager] io_manager) except +
