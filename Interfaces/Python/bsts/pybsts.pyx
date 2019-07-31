from model_manager cimport ScalarModelManager, ScalarManagedModel, PythonListIoManager, \
    ScalarStateSpaceSpecification, ModelOptions, LocalTrendSpecification, Matrix, Vector, \
    OdaOptions, SeasonSpecification, DoublePrior, PriorSpecification, \
    HierarchicalModelSpecification, move, SpdMatrix, seed_rng_externally, seed_rng_with_timestamp
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
    xdim = 0

    def __cinit__(self, family, specification, options):
        if not specification:
            raise ValueError("Can not create BSTS object with empty specification")

        if "predictor_names" in specification:
            self.xdim = len(specification["predictor_names"])

        self.cpp_scalar_model_manager.reset(ScalarModelManager.Create(_bytes(family), self.xdim))
        self.cpp_io_manager.reset(new PythonListIoManager())
        cdef vector[SeasonSpecification] seasons
        cdef unique_ptr[OdaOptions] oda_options
        cdef unique_ptr[LocalTrendSpecification] local_trend
        cdef unique_ptr[PriorSpecification] initial_state_prior
        cdef unique_ptr[PriorSpecification] sigma_prior
        cdef unique_ptr[PriorSpecification] predictors_prior
        cdef unique_ptr[PriorSpecification] trend_prior
        cdef unique_ptr[PriorSpecification] slope_prior
        cdef unique_ptr[PriorSpecification] slope_bias_prior
        cdef unique_ptr[DoublePrior] trend_df_prior
        cdef unique_ptr[DoublePrior] slope_df_prior
        cdef unique_ptr[PriorSpecification] slope_ar1_prior
        cdef unique_ptr[PriorSpecification] ar_prior
        cdef vector[string] predictor_names
        cdef ModelOptions *model_options
        cdef unique_ptr[HierarchicalModelSpecification] hierarchical_regression_specification
        cdef Vector prior_inclusion_probabilities
        cdef Vector prior_mean
        cdef SpdMatrix prior_precision

        bma_method = "SSVS"
        if family in ["gaussian"]:
            bma_method = "ODA"
        else:
            raise ValueError("Currently supported only 'gaussian' as a family.")

        sdy = 0.0
        if "sigma_prior" in specification and specification["sigma_prior"]:
            if type(specification["sigma_prior"]) == dict:
                sigma_upper_limit = specification["sigma_prior"]["sigma_upper_limit"]
                prior_df = specification["sigma_prior"]["prior_df"]
                sigma_guess = specification["sigma_prior"]["sigma_guess"]
                sdy = sigma_guess
                sigma_prior.reset(new PriorSpecification(
                    Vector(), # prior_inclusion_probabilities
                    Vector(), # prior_mean
                    SpdMatrix(), # prior_precision
                    Vector(), # prior_variance_diagonal
                    0, # max_flips
                    0.0, # initial_value
                    0.0, # mu
                    prior_df, # prior_df
                    sigma_guess, # prior_guess
                    sigma_guess, # sigma_guess
                    sigma_upper_limit, # sigma_upper_limit
                    False, # truncate
                    False, # positive
                    False)) # fixed
            else:
                sigma_upper_limit = 1.2 * specification["sigma_prior"]
                prior_df = 0.01
                sigma_guess = specification["sigma_prior"]
                sdy = sigma_guess
                sigma_prior.reset(new PriorSpecification(
                    Vector(), # prior_inclusion_probabilities
                    Vector(), # prior_mean
                    SpdMatrix(), # prior_precision
                    Vector(), # prior_variance_diagonal
                    0, # max_flips
                    0.0, # initial_value
                    0.0, # mu
                    prior_df, # prior_df
                    sigma_guess, # prior_guess
                    sigma_guess, # sigma_guess
                    sigma_upper_limit, # sigma_upper_limit
                    False, # truncate
                    False, # positive
                    False)) # fixed
        else:
            raise ValueError("Sigma prior is needed for 'gaussian' family.")

        if "seasons" in specification and type(specification["seasons"]==list) and len(specification["seasons"]) > 0:
            seasons.resize(len(specification["seasons"]))
            for season in specification["seasons"]:
                if "duration" in season and "number_of_seasons" in season:
                    seasons.push_back(deref(
                        new SeasonSpecification(<int>specification["number_of_seasons"],
                                                <int>specification["duration"])))

        if "bma_method" in specification:
            if specification["bma_method"].lower() == "oda":
                bma_method = "ODA"
                if "oda_options" in specification:
                    if "eigenvalue_fudge_factor" in specification["oda_options"] \
                        and "fallback_probability" in specification["oda_options"]:
                        oda_options.reset(new OdaOptions(<double>specification["eigenvalue_fudge_factor"],
                                                         <double>specification["fallback_probability"]))
                    else:
                        oda_options.reset(new OdaOptions(0.01, 0.0))
            elif specification["bma_method"].lower() == "ssvs":
                bma_method = "SSVS"

        if "local_trend" in specification:
            if "static_intercept" in specification["local_trend"] and specification["local_trend"]["static_intercept"]:
                if "initial_value" in specification and "sigma_prior" in specification:
                    initial_state_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        specification["initial_value"], # mu
                        0.0, # prior_df
                        0.0, # prior_guess
                        specification["sigma_prior"], # sigma_guess
                        1.0, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    local_trend.reset(new LocalTrendSpecification(move(trend_prior),
                        move(slope_prior),
                        move(slope_bias_prior),
                        move(trend_df_prior),
                        move(slope_df_prior),
                        move(slope_ar1_prior),
                        True, False))
                else:
                    raise ValueError("Static intercept needs 'initial_value' and 'sigma_prior' values provided")
            else:
                if  "local_level" in specification["local_trend"] and specification["local_trend"]["local_level"]:
                    prior_guess = sdy * 0.01
                    prior_df = 0.01
                    sigma_upper_limit = sdy
                    initial_value = specification["initial_value"]
                    mu = specification["initial_value"]
                    sigma_guess = sdy
                    trend_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        prior_df, # prior_df
                        prior_guess, # prior_guess
                        sigma_guess, # sigma_guess
                        sigma_upper_limit, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    local_trend.reset(new LocalTrendSpecification(move(trend_prior),
                        move(slope_prior),
                        move(slope_bias_prior),
                        move(trend_df_prior),
                        move(slope_df_prior),
                        move(slope_ar1_prior),
                        True, False))
                elif "local_linear_trend" in specification["local_trend"] and specification["local_trend"]["local_linear_trend"]:
                    prior_guess = sdy * 0.01
                    prior_df = 0.01
                    sigma_upper_limit = sdy
                    mu = specification["initial_value"]
                    sigma_guess = sdy
                    trend_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        prior_df, # prior_df
                        prior_guess, # prior_guess
                        sigma_guess, # sigma_guess
                        sigma_upper_limit, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    mu = 0.0
                    if "final_value" in specification and "number_of_time_points" in specification:
                        mu = (specification["final_value"] - specification["initial_value"])/specification["number_of_time_points"]
                    slope_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        prior_df, # prior_df
                        prior_guess, # prior_guess
                        sigma_guess, # sigma_guess
                        sigma_upper_limit, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    local_trend.reset(new LocalTrendSpecification(move(trend_prior),
                        move(slope_prior),
                        move(slope_bias_prior),
                        move(trend_df_prior),
                        move(slope_df_prior),
                        move(slope_ar1_prior),
                        True, False))
                elif "semilocal_linear_trend" in specification["local_trend"] and specification["local_trend"]["semilocal_linear_trend"]:
                    prior_guess = sdy * 0.01
                    prior_df = 0.01
                    sigma_upper_limit = sdy
                    mu = specification["initial_value"]
                    sigma_guess = sdy
                    trend_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        prior_df, # prior_df
                        prior_guess, # prior_guess
                        sigma_guess, # sigma_guess
                        sigma_upper_limit, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    mu = 0.0
                    slope_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        prior_df, # prior_df
                        prior_guess, # prior_guess
                        sigma_guess, # sigma_guess
                        sigma_upper_limit, # sigma_upper_limit
                        False, # truncate
                        False, # positive
                        False)) # fixed
                    mu = 0.0
                    sigma_guess = 1.0
                    truncate = True
                    positive = False
                    slope_ar1_prior.reset(new PriorSpecification(
                        Vector(), # prior_inclusion_probabilities
                        Vector(), # prior_mean
                        SpdMatrix(), # prior_precision
                        Vector(), # prior_variance_diagonal
                        0, # max_flips
                        0.0, # initial_value
                        mu, # mu
                        0.0, # prior_df
                        0.0, # prior_guess
                        sigma_guess, # sigma_guess
                        0.0, # sigma_upper_limit
                        truncate, # truncate
                        positive, # positive
                        False)) # fixed
                    local_trend.reset(new LocalTrendSpecification(move(trend_prior),
                        move(slope_prior),
                        move(slope_bias_prior),
                        move(trend_df_prior),
                        move(slope_df_prior),
                        move(slope_ar1_prior),
                        True, False))

        ar_order = 0
        if "ar_order" in specification and specification["ar_order"]:
            ar_order = specification["ar_order"]
            if "ar_prior" in specification and specification["ar_prior"]:
                prior_inclusion_probabilities = Vector()
                prior_mean = Vector()
                prior_precision = SpdMatrix()
                max_flips = 10
                prior_df = 0.01
                prior_guess = 0.01*sdy
                sigma_upper_limit = sdy
                truncate = False
                if "inclusion_probabilities" in specification["ar_prior"]:
                    prior_inclusion_probabilities = nparray_to_vector(specification["ar_prior"]["inclusion_probabilities"])
                if "mean" in specification["ar_prior"]:
                    if type(specification["ar_prior"]["mean"]) == list:
                        prior_mean = nparray_to_vector(specification["ar_prior"]["mean"])
                    else:
                        prior_mean = Vector(<double>specification["ar_prior"]["mean"], <int>specification["ar_order"])
                if "precision" in specification["ar_prior"]:
                    prior_precision = SpdMatrix(nparray_to_matrix(specification["precision"]), True)
                if "max_flips" in specification["ar_prior"]:
                    max_flips = specification["ar_prior"]["max_flips"]
                if "prior_df" in specification["ar_prior"]:
                    prior_df = specification["ar_prior"]["prior_df"]
                if "prior_guess" in specification["ar_prior"]:
                    prior_guess = specification["ar_prior"]["prior_guess"]
                if "sigma_upper_limit" in specification:
                    sigma_upper_limit = specification["ar_prior"]["sigma_upper_limit"]
                if "truncate" in specification and specification["ar_prior"]["truncate"]:
                    truncate = True
                ar_prior.reset(new PriorSpecification(
                    prior_inclusion_probabilities, # prior_inclusion_probabilities
                    prior_mean, # prior_mean
                    prior_precision, # prior_precision
                    Vector(), # prior_variance_diagonal
                    max_flips, # max_flips
                    0.0, # initial_value
                    0.0, # mu
                    prior_df, # prior_df
                    prior_guess, # prior_guess
                    1.0, # sigma_guess
                    sigma_upper_limit, # sigma_upper_limit
                    truncate, # truncate
                    False, # positive
                    False)) # fixed

        dynamic_regression = False
        if "dynamic_regression" in specification:
            dynamic_regression = specification["dynamic_regression"]

        if "predictor_names" in specification and type(specification["predictor_names"])==list \
                and len(specification["predictor_names"]) > 0:
            predictor_names.resize(len(specification["predictor_names"]))
            for name in specification["predictor_names"]:
                predictor_names.push_back(_bytes(name))

        save_state_contributions = True
        save_full_state = True
        save_prediction_errors = True
        niter = 100
        ping = niter / 10
        burn = 10
        forecast_horizon = 1
        timeout_threshold_seconds = 10
        if options:
            if "save_state_contributions" in options and options["save_state_contributions"]:
                save_state_contributions = True
            if "save_full_state" in options and options["save_full_state"]:
                save_full_state = True
            if "save_prediction_errors" in options and options["save_prediction_errors"]:
                save_prediction_errors = True
            if "niter" in options and options["niter"] > 0:
                niter = <int>(options["niter"])
            if "ping" in options and options["ping"] > 0:
                ping = <int>(options["ping"])
            if "burn" in options and options["burn"] > 0:
                burn = <int>(options["burn"])
            if "forecast_horizon" in options and options["forecast_horizon"] > 0:
                forecast_horizon = <int>(options["forecast_horizon"])
            if "timeout_threshold_seconds" in options and options["timeout_threshold_seconds"] > 0:
                timeout_threshold_seconds = <double>(options["timeout_threshold_seconds"])
        model_options = new ModelOptions(
            save_state_contributions,
            save_full_state,
            save_prediction_errors,
            niter,
            ping,
            burn,
            forecast_horizon,
            timeout_threshold_seconds
            )

        self.cpp_model.reset(deref(self.cpp_scalar_model_manager).CreateModel(
            new ScalarStateSpaceSpecification(
                move(initial_state_prior),
                move(sigma_prior),
                move(predictors_prior),
                move(local_trend),
                _bytes(bma_method), move(oda_options),
                seasons,
                move(hierarchical_regression_specification),
                move(ar_prior),
                predictor_names,
                ar_order, dynamic_regression),
            model_options,
            self.cpp_io_manager))

    def crepr(self):
        return <bytes>deref(self.cpp_io_manager).repr().c_str()

    def fit(self, arg1, arg2=None, seed=None):
        cdef Matrix empty = Matrix()
        status = False
        if seed is None:
            seed_rng_with_timestamp()
        else:
            seed_rng_externally(<int>seed)
        if arg2 is None:
            y = arg1
            if y is None or len(y) == 0:
                raise ValueError("BSTS.fit can not be called with empty data.")
            status = deref(self.cpp_model).fit(nparray_to_vector(y), empty)
        else:
            X = arg1
            y = arg2
            if X is None or len(X) == 0 or len(y) == 0:
                raise ValueError("BSTS.fit can not be called with empty data.")
            status = deref(self.cpp_model).fit(nparray_to_vector(y), nparray_to_matrix(X))
        if not status:
            raise Exception("Failure during model fit.")

    def predict(self, X=None, indices=[], seed=None):
        cdef Matrix m = Matrix()
        cdef vector[int] prediction_indices = <vector[int]>indices
        ret = np.empty((0, 0), dtype=np.float64)
        if seed is None:
            seed_rng_with_timestamp()
        else:
            seed_rng_externally(<int>seed)
        if self.xdim > 0:
            if X is None or len(X) == 0:
                ValueError("BSTS.predict can not be called with empty data.")
            m = nparray_to_matrix(X)
        if indices:
            ret = matrix_to_nparray(deref(self.cpp_model).predict(m, prediction_indices))
        else:
            ret = matrix_to_nparray(deref(self.cpp_model).predict(m))
        return ret

    def seed(s):
        seed_rng_externally(s)

