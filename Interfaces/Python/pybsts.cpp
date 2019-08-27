#include <pybind11/pybind11.h>

#include "Models/GaussianModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/StaticInterceptStateModel.hpp"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>);

namespace BOOM {
namespace pybsts {


PYBIND11_MODULE(pybsts, m) {
    m.doc() = "pybind11 pybsts plugin"; // optional module docstring

    // Vector
    py::class_<Vector>(m, "Vector")
        .def("__getitem__",
                [](Vector &v, int i) -> double & {
                    return v[i];
                },
                py::return_value_policy::reference_internal // ref + keepalive
            )
        .def("__iter__",
                [](Vector &v) {
                   return py::make_iterator<
                         py::return_value_policy::copy, Vector::iterator, Vector::iterator, double>(
                        v.begin(), v.end());
                },
                py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
            );

    // Matrix
    py::class_<Matrix>(m, "Matrix")
        .def("size", &Matrix::size)
        .def("nrow", &Matrix::nrow)
        .def("ncol", &Matrix::ncol)
        .def("__getitem__",
                [](Matrix &m, int i, int j) -> double & {
                    return m(i, j);
                },
                py::return_value_policy::reference_internal // ref + keepalive
            )
        .def("__iter__",
                [](Matrix &m) {
                   return py::make_iterator<
                         py::return_value_policy::copy, Matrix::dVector::iterator, Matrix::dVector::iterator, double>(
                         m.begin(), m.end());
                },
                py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
            );

    // Data
    py::class_<Data, Ptr<Data> >(m, "Data");
    py::class_<DoubleData, Ptr<DoubleData>, Data>(m, "DoubleData")
        .def(py::init<double>());
    py::class_<StateSpace::MultiplexedData, Ptr<StateSpace::MultiplexedData>, Data>(m, "MultiplexedData");
    py::class_<StateSpace::MultiplexedDoubleData, Ptr<StateSpace::MultiplexedDoubleData>, StateSpace::MultiplexedData>(m, "MultiplexedDoubleData")
        .def(py::init<double>())
        .def("add_data", &StateSpace::MultiplexedDoubleData::add_data);

    // DataPolicy
    py::class_<IID_DataPolicy<StateSpace::MultiplexedDoubleData>>(m, "MultiplexedDoubleDataPolicy");

    // Models
    py::class_<GammaModelBase, Ptr<GammaModelBase> >(m, "GammaModelBase");

    py::class_<ChisqModel, Ptr<ChisqModel>, GammaModelBase>(m, "ChisqModel")
        .def(py::init<double, double>())
        .def("sample_posterior", &ChisqModel::sample_posterior)
        .def_property("df", &ChisqModel::df, &ChisqModel::set_df)
        .def_property("sigma_estimate", &ChisqModel::sigma, &ChisqModel::set_sigma_estimate)
        .def("alpha", &ChisqModel::alpha)
        .def("beta", &ChisqModel::beta)
        .def("sum_of_squares", &ChisqModel::sum_of_squares);

    py::class_<GaussianModelBase, Ptr<GaussianModelBase> >(m, "GaussianModelBase");

    py::class_<GaussianModel, Ptr<GaussianModel>, GaussianModelBase>(m, "GaussianModel")
        .def(py::init<double, double>())
        .def("set_method", &GaussianModel::set_method)
        .def("sample_posterior", &GaussianModel::sample_posterior)
        .def_property("mu", &GaussianModel::mu, &GaussianModel::set_mu)
        .def_property("sigsq", &GaussianModel::sigsq, &GaussianModel::set_sigsq);

    py::class_<ZeroMeanGaussianModel, Ptr<ZeroMeanGaussianModel>, GaussianModelBase>(m, "ZeroMeanGaussianModel")
        .def(py::init<>())
        .def("set_method", &ZeroMeanGaussianModel::set_method)
        .def("sample_posterior", &ZeroMeanGaussianModel::sample_posterior)
        .def_property("sigsq", &ZeroMeanGaussianModel::sigsq, &ZeroMeanGaussianModel::set_sigsq);

    py::class_<StateSpaceModelBase, Ptr<StateSpaceModelBase> >(m, "StateSpaceModelBase");

    py::class_<StateSpaceModel, Ptr<StateSpaceModel>, StateSpaceModelBase>(m, "StateSpaceModel")
        .def(py::init<>())
        .def("set_method", &StateSpaceModel::set_method)
        .def("sample_posterior", &StateSpaceModel::sample_posterior)
        .def("add_data", (void (IID_DataPolicy<StateSpace::MultiplexedDoubleData>::*)(StateSpace::MultiplexedDoubleData *)) &IID_DataPolicy<StateSpace::MultiplexedDoubleData>::add_data)
        .def("add_state", &StateSpaceModelBase::add_state)
        .def("forecast", &StateSpaceModel::forecast)
        .def("observation_model", (ZeroMeanGaussianModel *(StateSpaceModel::*)()) &StateSpaceModel::observation_model, 
             "Get pointer to observation model", py::return_value_policy::reference_internal);

    py::class_<StateModelBase, Ptr<StateModelBase> >(m, "StateModelBase");
    py::class_<StateModel, Ptr<StateModel>, StateModelBase>(m, "StateModel");
    py::class_<StaticInterceptStateModel, Ptr<StaticInterceptStateModel>, StateModel>(m, "StaticInterceptStateModel")
        .def(py::init<>())
        .def("set_initial_state_mean", &StaticInterceptStateModel::set_initial_state_mean)
        .def("set_initial_state_variance", &StaticInterceptStateModel::set_initial_state_variance);

    // Samplers
    py::class_<PosteriorSampler, Ptr<PosteriorSampler> >(m, "PosteriorSampler");

    py::class_<StateSpacePosteriorSampler, Ptr<StateSpacePosteriorSampler>, PosteriorSampler>(m, "StateSpacePosteriorSampler")
        .def(py::init<StateSpaceModelBase *>());

    py::class_<ZeroMeanGaussianConjSampler, Ptr<ZeroMeanGaussianConjSampler>, PosteriorSampler>(m, "ZeroMeanGaussianConjSampler")
        .def(py::init<ZeroMeanGaussianModel *, Ptr<GammaModelBase> &>());

    }
}
}

