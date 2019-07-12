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

StateSpaceModel * StateSpaceModelManager::CreateObservationModel(const ScalarStateSpaceSpecification *specification) {
  model_.reset(new StateSpaceModel);

  if (specification->sigma_prior()) {
    ZeroMeanGaussianModel *observation_model = model_->observation_model();
    Ptr<ZeroMeanGaussianConjSampler> sigma_sampler(
        new ZeroMeanGaussianConjSampler(
            observation_model,
            specification->sigma_prior()->prior_df(),
            specification->sigma_prior()->sigma_guess()));
    sigma_sampler->set_sigma_upper_limit(specification->sigma_prior()->sigma_upper_limit());
    observation_model->set_method(sigma_sampler);

    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(model_.get()));

    model_->set_method(sampler);
  } else {
    report_error("Empty sigma_prior in StateSpaceModelManager::CreateObservationModel.");
  }

  return model_.get();
}

HoldoutErrorSampler StateSpaceModelManager::CreateHoldoutSampler(
    const ScalarStateSpaceSpecification *specification,
    const PyBstsOptions *options,
    const Vector& response,
    const Matrix& inputdata,
    const std::vector<bool> &response_is_observed,
    int cutpoint,
    int niter,
    bool standardize,
    Matrix *prediction_error_output) {
  Ptr<StateSpaceModel> model = static_cast<StateSpaceModel *>(CreateModel(specification, options));
  AddData(response, response_is_observed);

  std::vector<Ptr<StateSpace::MultiplexedDoubleData>> data = model->dat();
  model_->clear_data();
  for (int i = 0; i <= cutpoint; ++i) {
    model_->add_data(data[i]);
  }
  Vector holdout_data;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    Ptr<StateSpace::MultiplexedDoubleData> data_point = data[i];
    for (int j = 0; j < data[i]->total_sample_size(); ++j) {
      holdout_data.push_back(data[i]->double_data(j).value());
    }
  }
  return HoldoutErrorSampler(new StateSpaceModelPredictionErrorSampler(
      model, holdout_data, niter, standardize, prediction_error_output));
}

StateSpaceModelPredictionErrorSampler::StateSpaceModelPredictionErrorSampler(
    const Ptr<StateSpaceModel> &model,
    const Vector &holdout_data,
    int niter,
    bool standardize,
    Matrix *errors)
    : model_(model),
      holdout_data_(holdout_data),
      niter_(niter),
      standardize_(standardize),
      errors_(errors)
{}

Vector StateSpaceModelManager::SimulateForecast(const Vector &final_state) {
  return model_->simulate_forecast(rng(), forecast_horizon_, final_state);
}

void StateSpaceModelPredictionErrorSampler::sample_holdout_prediction_errors() {
  model_->sample_posterior();
  errors_->resize(niter_, model_->time_dimension() + holdout_data_.size());
  for (int i = 0; i < niter_; ++i) {
    model_->sample_posterior();
    Vector error_sim = model_->one_step_prediction_errors();
    error_sim.concat(model_->one_step_holdout_prediction_errors(
        holdout_data_, model_->final_state(), standardize_));
    errors_->row(i) = error_sim;
  }
}

void StateSpaceModelManager::AddData(
    const Vector &response,
    const std::vector<bool> &response_is_observed) {
  if (!response_is_observed.empty()
      && (response.size() != response_is_observed.size())) {
    report_error("Vectors do not match in StateSpaceModelManager::AddData.");
  }
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
    model_->add_data(data[i]);
  }
}

}  // namespace pybsts
}  // namespace BOOM