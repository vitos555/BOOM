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

#ifndef PYBSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_
#define PYBSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_

#include "state_space_gaussian_model_manager.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"

namespace BOOM {
namespace pybsts {

class StateSpaceRegressionHoldoutErrorSampler
    : public HoldoutErrorSamplerImpl {
 public:
  // Args:
  //   model:  The model containing data up to a specified cutpoint.
  //   holdout_responses:  Observed values after the cutpoint.
  //   holdout_predictors: A matrix of observed predictors corresponding to
  //     holdout_responses.
  //   niter: The desired number of draws (MCMC iterations) from the posterior
  //     distribution.
  //   errors:  A matrix that will hold the output of the simulation.
  StateSpaceRegressionHoldoutErrorSampler(
      const Ptr<StateSpaceRegressionModel> &model,
      const Vector &holdout_responses,
      const Matrix &holdout_predictors,
      int niter,
      bool standardize,
      Matrix *errors)
      : model_(model),
      holdout_responses_(holdout_responses),
      holdout_predictors_(holdout_predictors),
      niter_(niter),
      standardize_(standardize),
      errors_(errors) {}

  void sample_holdout_prediction_errors() override;

 private:
  Ptr<StateSpaceRegressionModel> model_;
  Vector holdout_responses_;
  Matrix holdout_predictors_;
  int niter_;
  bool standardize_;
  Matrix *errors_;
};

class StateSpaceRegressionModelManager
    : public GaussianModelManagerBase {
 public:
  explicit StateSpaceRegressionModelManager(int predictor_dimension);

  StateSpaceRegressionModel * CreateObservationModel(const ScalarStateSpaceSpecification *specification) override;

  HoldoutErrorSampler CreateHoldoutSampler(
    const ScalarStateSpaceSpecification *specification,
    const PyBstsOptions *options,
    const Vector& responses,
    const Matrix& predictors,
    const std::vector<bool> &response_is_observed,
    int cutpoint,
    int niter,
    bool standardize,
    Matrix *prediction_error_output) override;
  
  Vector SimulateForecast(const Vector &final_state) override;

 private:
  void SetSsvsRegressionSampler(const ScalarStateSpaceSpecification *specification);
  void SetOdaRegressionSampler(const ScalarStateSpaceSpecification *specification);
  void DropUnforcedCoefficients(const Ptr<GlmModel> &glm, const BOOM::Vector &prior_inclusion_probs);  
  void AddData(
    const Vector &response,
    const Matrix &predictors,
    const std::vector<bool> &response_is_observed);

  Ptr<StateSpaceRegressionModel> model_;
  int predictor_dimension_;
  Matrix forecast_predictors_;
};

}  // namespace pybsts
}  // namespace BOOM

#endif  // PYBSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_
