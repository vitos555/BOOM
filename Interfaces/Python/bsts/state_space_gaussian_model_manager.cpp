// Copyright 2019 UMG. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include "state_space_gaussian_model_manager.hpp"
#include "model_manager.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

namespace BOOM {
namespace pybsts {

void GaussianModelManagerBase::init_io_manager(
            ScalarStateSpaceModelBase* sampling_model,
            ModelOptions *options,
            std::shared_ptr<PythonListIoManager> io_manager) {
  // It is only possible to compute log likelihood for Gaussian models.
  io_manager->add_list_element(
      new BOOM::NativeUnivariateListElement(
          new LogLikelihoodCallback(sampling_model),
          "log.likelihood",
          nullptr));
}

ScalarManagedModel* StateSpaceModelManager::CreateModel(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    std::shared_ptr<PythonListIoManager> io_manager) {
  StateSpaceManagedModel* model = nullptr;
  if (specification) {
    ScalarStateSpaceModelBase* sampling_model = CreateObservationModel(specification, io_manager);
    if (!sampling_model) {
      report_error("Error while creating an sampling model.");
    } else {
      model = new StateSpaceManagedModel(specification, options, sampling_model, io_manager);
    }
    init_io_manager(sampling_model, options, io_manager);
  } else {
    report_error("Empty specification in StateSpaceManagedModel::CreateModel.");
  }
  return model;
}

StateSpaceModel * StateSpaceModelManager::CreateObservationModel(const ScalarStateSpaceSpecification *specification,
    std::shared_ptr<PythonListIoManager> io_manager) {
  StateSpaceModel *sampling_model = new StateSpaceModel;

  if (specification->sigma_prior()) {
    std::ostringstream msg;
    msg << "StateSpaceModelManager::CreateObservationModel: prior_df:" << std::setprecision(22) << specification->sigma_prior()->prior_df();
    msg << ", sigma_guess: " << specification->sigma_prior()->sigma_guess();
    msg << ", sigma_upper_limit: " << specification->sigma_prior()->sigma_upper_limit() << std::endl;
    report_message(msg.str());
    ZeroMeanGaussianModel *observation_model = sampling_model->observation_model();
    Ptr<ZeroMeanGaussianConjSampler> sigma_sampler(
        new ZeroMeanGaussianConjSampler(
            observation_model,
            specification->sigma_prior()->prior_df(),
            specification->sigma_prior()->sigma_guess()));
    sigma_sampler->set_sigma_upper_limit(specification->sigma_prior()->sigma_upper_limit());
    observation_model->set_method(sigma_sampler);

    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(sampling_model));

    sampling_model->set_method(sampler);
  } else {
    report_error("Empty sigma_prior in StateSpaceModelManager::CreateObservationModel.");
  }

  // Make the io_manager aware of the model parameters.
  io_manager->add_list_element(new StandardDeviationListElement(
      sampling_model->observation_model()->Sigsq_prm(),
      "sigma.obs"));

  return sampling_model;
}

HoldoutErrorSampler StateSpaceModelManager::CreateHoldoutSampler(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    const Vector& response,
    const Matrix& inputdata,
    const std::vector<bool> &response_is_observed,
    int cutpoint,
    int niter,
    bool standardize,
    Matrix *prediction_error_output) {
  std::shared_ptr<PythonListIoManager> io_manager(new PythonListIoManager());
  std::unique_ptr<StateSpaceManagedModel> model;
  model.reset(static_cast<StateSpaceManagedModel*>(CreateModel(specification, options, io_manager)));
  model->AddData(response, response_is_observed);
  StateSpaceModel *sampling_model = static_cast<StateSpaceModel*>(model->sampling_model());

  std::vector<Ptr<StateSpace::MultiplexedDoubleData>> data = sampling_model->dat();
  sampling_model->clear_data();
  for (int i = 0; i <= cutpoint; ++i) {
    sampling_model->add_data(data[i]);
  }
  Vector holdout_data;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    Ptr<StateSpace::MultiplexedDoubleData> data_point = data[i];
    for (int j = 0; j < data[i]->total_sample_size(); ++j) {
      holdout_data.push_back(data[i]->double_data(j).value());
    }
  }
  return HoldoutErrorSampler(new StateSpaceModelPredictionErrorSampler(
      std::move(model), holdout_data, niter, standardize, prediction_error_output));
}

StateSpaceModelPredictionErrorSampler::StateSpaceModelPredictionErrorSampler(
    std::unique_ptr<StateSpaceManagedModel> model,
    const Vector &holdout_data,
    int niter,
    bool standardize,
    Matrix *errors)
    : model_(std::move(model)), 
      holdout_data_(holdout_data),
      niter_(niter),
      standardize_(standardize),
      errors_(errors)
{ }

void StateSpaceModelPredictionErrorSampler::sample_holdout_prediction_errors() {
  StateSpaceModel *sampling_model = static_cast<StateSpaceModel*>(model_->sampling_model());
  sampling_model->sample_posterior();
  errors_->resize(niter_, sampling_model->time_dimension() + holdout_data_.size());
  for (int i = 0; i < niter_; ++i) {
    sampling_model->sample_posterior();
    Vector error_sim = sampling_model->one_step_prediction_errors();
    error_sim.concat(sampling_model->one_step_holdout_prediction_errors(
        holdout_data_, sampling_model->final_state(), standardize_));
    errors_->row(i) = error_sim;
  }
}

StateSpaceManagedModel::StateSpaceManagedModel(const ScalarStateSpaceSpecification *specification,
            ModelOptions* options,
            ScalarStateSpaceModelBase* sampling_model,
            std::shared_ptr<PythonListIoManager> io_manager) :
    ScalarManagedModel(specification, options, sampling_model, io_manager)
{}

void StateSpaceManagedModel::AddData(
    const Vector &response,
    const std::vector<bool> &response_is_observed
    ) {
  if (!response_is_observed.empty()
      && (response.size() != response_is_observed.size())) {
    report_error("Vectors do not match in StateSpaceModelManager::AddData.");
  }
  StateSpaceModel *sampling_model = static_cast<StateSpaceModel*>(this->sampling_model());
  std::vector<Ptr<StateSpace::MultiplexedDoubleData>> data;
  data.reserve(NumberOfTimePoints());
  for (int i = 0; i < NumberOfTimePoints(); ++i) {
    data.push_back(new StateSpace::MultiplexedDoubleData);
  }
  for (int i = 0; i < response.size(); ++i) {
    NEW(DoubleData, observation)(response[i]);
    if (!response_is_observed.empty() && !response_is_observed[i]) {
      observation->set_missing_status(Data::completely_missing);
    }
    data[TimestampMapping(i)]->add_data(observation);
  }
  for (int i = 0; i < NumberOfTimePoints(); ++i) {
    if (data[i]->all_missing()) {
      data[i]->set_missing_status(Data::completely_missing);
    }
    sampling_model->add_data(data[i]);
  }

}

void StateSpaceManagedModel::AddData(
        const Vector &response,
        const Matrix &predictors,
        const std::vector<bool> &response_is_observed)
{
  report_error("Wrong AddData method is used.");
}

Vector StateSpaceManagedModel::SimulateForecast() {
  StateSpaceModel *sampling_model = static_cast<StateSpaceModel*>(this->sampling_model());
  return sampling_model->simulate_forecast(rng(), this->options()->forecast_horizon(), final_state());
}

void StateSpaceManagedModel::sample_posterior() {
  StateSpaceModel *sampling_model = static_cast<StateSpaceModel*>(this->sampling_model());
  sampling_model->sample_posterior();
}

}  // namespace pybsts
}  // namespace BOOM