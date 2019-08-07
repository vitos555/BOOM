// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "state_space_regression_model_manager.hpp"
#include "state_space_gaussian_model_manager.hpp"

#include "prior_specification.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
namespace pybsts {

void DropUnforcedCoefficients(const Ptr<GlmModel> &glm,
                              const BOOM::Vector &prior_inclusion_probs) {
  glm->coef().drop_all();
  for (int i = 0; i < prior_inclusion_probs.size(); ++i) {
    if (prior_inclusion_probs[i] >= 1.0) {
      glm->coef().add(i);
    }
  }
}

StateSpaceRegressionHoldoutErrorSampler::StateSpaceRegressionHoldoutErrorSampler(
    std::unique_ptr<ScalarManagedModel> model,
    const Vector &holdout_responses,
    const Matrix &holdout_predictors,
    int niter,
    bool standardize,
    Matrix *errors) :
      model_(std::move(model)),
      holdout_responses_(holdout_responses),
      holdout_predictors_(holdout_predictors),
      niter_(niter),
      standardize_(standardize),
      errors_(errors)
{ }  

StateSpaceRegressionModelManager::StateSpaceRegressionModelManager(int predictor_dimension)
    : predictor_dimension_(predictor_dimension) {}

ScalarManagedModel* StateSpaceRegressionModelManager::CreateModel(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    std::shared_ptr<PythonListIoManager> io_manager) {
  StateSpaceRegressionManagedModel* model = nullptr;
  if (specification) {
    ScalarStateSpaceModelBase* sampling_model = CreateObservationModel(specification, io_manager);
    if (!sampling_model) {
      report_error("Error while creating an sampling model.");
    } else {
      model = new StateSpaceRegressionManagedModel(specification, options, sampling_model, io_manager);
    }
    init_io_manager(sampling_model, options, io_manager);  
  } else {
    report_error("Empty specification in StateSpaceRegressionManagedModel::CreateModel.");
  }
  return model;
}

StateSpaceRegressionModel * StateSpaceRegressionModelManager::CreateObservationModel(
    const ScalarStateSpaceSpecification *specification,
    std::shared_ptr<PythonListIoManager> io_manager) {
   if (predictor_dimension_ < 0) {
    report_error("Negative value of predictor_dimension_ in "
                 "CreateObservationModel.");
  }
  StateSpaceRegressionModel *sampling_model = new StateSpaceRegressionModel(predictor_dimension_);

  // A NULL priors signals that no posterior sampler is needed.
  if (specification->predictors_prior()) {
    if (specification->bma_method() == "SSVS") {
      SetSsvsRegressionSampler(specification, sampling_model);
    } else if (specification->bma_method() == "ODA") {
      SetOdaRegressionSampler(specification, sampling_model);
    } else {
      std::ostringstream err;
      err << "Unrecognized value of bma_method: " << specification->bma_method();
      report_error(err.str());
    }
    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(sampling_model));
    sampling_model->set_method(sampler);
  }

  Ptr<RegressionModel> regression(sampling_model->regression_model());
  io_manager->add_list_element(
      new GlmCoefsListElement(regression->coef_prm(), "coefficients"));
  io_manager->add_list_element(
      new StandardDeviationListElement(regression->Sigsq_prm(),
                                       "sigma.obs"));
  return sampling_model;
}

Vector StateSpaceRegressionManagedModel::SimulateForecast() {
  StateSpaceRegressionModel *sampling_model = static_cast<StateSpaceRegressionModel*>(this->sampling_model());
  if (ForecastTimestamps().empty()) {
    return sampling_model->simulate_forecast(rng(), forecast_predictors_, final_state());
  } else {
    return sampling_model->simulate_multiplex_forecast(rng(),
                                               forecast_predictors_,
                                               final_state(),
                                               ForecastTimestamps());
  }
}

void StateSpaceRegressionManagedModel::update_forecast_predictors(const Matrix &x, const std::vector<int> &forecast_timestamps) {
  ManagedModel::update_forecast_predictors(x, forecast_timestamps);
  forecast_predictors_ = x;
}

void StateSpaceRegressionManagedModel::sample_posterior() {
  StateSpaceRegressionModel *sampling_model = static_cast<StateSpaceRegressionModel*>(this->sampling_model());
  sampling_model->sample_posterior();
}

void StateSpaceRegressionModelManager::SetSsvsRegressionSampler(const ScalarStateSpaceSpecification *specification,
    const StateSpaceRegressionModel *sampling_model) {
  RegressionConjugateSpikeSlabPrior prior(
        specification->predictors_prior()->prior_inclusion_probabilities(),
        specification->predictors_prior()->prior_mean(), specification->predictors_prior()->prior_precision(),
        specification->predictors_prior()->prior_variance_diagonal(), specification->predictors_prior()->max_flips(),
        specification->predictors_prior()->prior_df(), specification->predictors_prior()->prior_guess(),
        specification->predictors_prior()->sigma_upper_limit(),
      sampling_model->regression_model()->Sigsq_prm());
  DropUnforcedCoefficients(sampling_model->regression_model(),
                           prior.prior_inclusion_probabilities());
  Ptr<BregVsSampler> sampler(new BregVsSampler(
      sampling_model->regression_model().get(),
      prior.slab(),
      prior.siginv_prior(),
      prior.spike()));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  if (prior.max_flips() > 0) {
    sampler->limit_model_selection(prior.max_flips());
  }
  sampling_model->regression_model()->set_method(sampler);
}

void StateSpaceRegressionModelManager::SetOdaRegressionSampler(const ScalarStateSpaceSpecification *specification,
    const StateSpaceRegressionModel *sampling_model) {
  IndependentRegressionSpikeSlabPrior prior(
          specification->predictors_prior()->prior_inclusion_probabilities(),
          specification->predictors_prior()->prior_mean(), specification->predictors_prior()->prior_variance_diagonal(),
          specification->predictors_prior()->max_flips(), specification->predictors_prior()->prior_df(),
          specification->predictors_prior()->sigma_guess(), specification->predictors_prior()->sigma_upper_limit(),
      sampling_model->regression_model()->Sigsq_prm());
  double eigenvalue_fudge_factor = 0.001;
  double fallback_probability = 0.0;
  if (specification->oda_options()) {
    eigenvalue_fudge_factor = specification->oda_options()->eigenvalue_fudge_factor();
    fallback_probability = specification->oda_options()->fallback_probability();
  }
  Ptr<SpikeSlabDaRegressionSampler> sampler(
      new SpikeSlabDaRegressionSampler(
          sampling_model->regression_model().get(),
          prior.slab(),
          prior.siginv_prior(),
          prior.prior_inclusion_probabilities(),
          eigenvalue_fudge_factor,
          fallback_probability));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  DropUnforcedCoefficients(sampling_model->regression_model(),
                           prior.prior_inclusion_probabilities());
  sampling_model->regression_model()->set_method(sampler);
}

StateSpaceRegressionManagedModel::StateSpaceRegressionManagedModel(const ScalarStateSpaceSpecification *specification,
            ModelOptions* options,
            ScalarStateSpaceModelBase* sampling_model,
            std::shared_ptr<PythonListIoManager> io_manager) :
    ScalarManagedModel(specification, options, sampling_model, io_manager)
{}

void StateSpaceRegressionManagedModel::AddData(
    const Vector &response,
    const std::vector<bool> &response_is_observed) {
  report_error("Wrong AddData method is used.");
}

void StateSpaceRegressionManagedModel::AddData(
    const Vector &response,
    const Matrix &predictors,
    const std::vector<bool> &response_is_observed) {
  StateSpaceRegressionModel *sampling_model = static_cast<StateSpaceRegressionModel*>(this->sampling_model());
  predictors_ = predictors;
  if (nrow(predictors) != response.size()
      || response_is_observed.size() != response.size()) {
    std::ostringstream err;
    err << "Argument sizes do not match in "
        << "StateSpaceRegressionModelManager::AddData" << endl
        << "nrow(predictors) = " << nrow(predictors) << endl
        << "response.size()  = " << response.size() << endl
        << "observed.size()  = " << response_is_observed.size();
    report_error(err.str());
  }

  for (int i = 0; i < response.size(); ++i) {
    Ptr<RegressionData> dp(new RegressionData(response[i], predictors.row(i)));
    if (!response_is_observed[i]) {
      dp->set_missing_status(Data::partly_missing);
    }
    sampling_model->add_regression_data(dp);
  }
}
void StateSpaceRegressionHoldoutErrorSampler::sample_holdout_prediction_errors() {

  StateSpaceRegressionModel *sampling_model = static_cast<StateSpaceRegressionModel*>(model_->sampling_model());
  sampling_model->sample_posterior();
  errors_->resize(niter_, sampling_model->time_dimension() + holdout_responses_.size());
  for (int i = 0; i < niter_; ++i) {
    sampling_model->sample_posterior();
    Vector all_errors = sampling_model->one_step_prediction_errors(standardize_);
    all_errors.concat(sampling_model->one_step_holdout_prediction_errors(
        holdout_predictors_, holdout_responses_, sampling_model->final_state(), standardize_));
    errors_->row(i) = all_errors;
  }
}

HoldoutErrorSampler StateSpaceRegressionModelManager::CreateHoldoutSampler(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    const Vector& responses,
    const Matrix& predictors,
    const std::vector<bool> &response_is_observed,
    int cutpoint,
    int niter,
    bool standardize,
    Matrix *prediction_error_output) {
  std::shared_ptr<PythonListIoManager> io_manager(new PythonListIoManager());
  std::unique_ptr<StateSpaceRegressionManagedModel> model;
  model.reset(static_cast<StateSpaceRegressionManagedModel*>(CreateModel(specification, options, io_manager)));
  model->AddData(responses, predictors, response_is_observed);

  StateSpaceRegressionModel *sampling_model = static_cast<StateSpaceRegressionModel*>(model->sampling_model());
  std::vector<Ptr<StateSpace::MultiplexedRegressionData>> data = sampling_model->dat();
  sampling_model->clear_data();
  for (int i = 0; i <= cutpoint; ++i) {
    sampling_model->add_multiplexed_data(data[i]);
  }
  int holdout_sample_size = 0;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    holdout_sample_size += data[i]->total_sample_size();
  }
  Matrix holdout_predictors(holdout_sample_size,
                            sampling_model->observation_model()->xdim());
  Vector holdout_response(holdout_sample_size);
  int index = 0;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    for (int j = 0; j < data[i]->total_sample_size(); ++j) {
      holdout_predictors.row(index) = data[i]->regression_data(j).x();
      holdout_response[index] = data[i]->regression_data(j).y();
      ++index;
    }
  }
  return HoldoutErrorSampler(new StateSpaceRegressionHoldoutErrorSampler(
      std::move(model), holdout_response, holdout_predictors,
      niter,
      standardize,
      prediction_error_output));
}

}  // namespace pybsts
}  // namespace BOOM
