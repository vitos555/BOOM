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
      std::unique_ptr<ScalarManagedModel> model,
      const Vector &holdout_responses,
      const Matrix &holdout_predictors,
      int niter,
      bool standardize,
      Matrix *errors);

  void sample_holdout_prediction_errors() override;

 private:
  std::unique_ptr<ScalarManagedModel> model_;
  Vector holdout_responses_;
  Matrix holdout_predictors_;
  int niter_;
  bool standardize_;
  Matrix *errors_;
};

class StateSpaceRegressionManagedModel : public ScalarManagedModel {
  public:
    StateSpaceRegressionManagedModel(const ScalarStateSpaceSpecification *specification,
            ModelOptions* options,
            ScalarStateSpaceModelBase* sampling_model,
            std::shared_ptr<PythonListIoManager> io_manager);
    Vector SimulateForecast() override;
    void AddData(
      const Vector &response,
      const Matrix &predictors,
      const std::vector<bool> &response_is_observed) override;
    void AddData(const Vector &response, const std::vector<bool> &response_is_observed) override;
    const Matrix& predictors() const { return predictors_; }
    void sample_posterior() override;

  protected:
    void update_forecast_predictors(const Matrix &x, const std::vector<int> &forecast_timestamps) override;

  private:
    Matrix forecast_predictors_;
    Matrix predictors_;
};

class StateSpaceRegressionModelManager
    : public GaussianModelManagerBase {
  public:
    explicit StateSpaceRegressionModelManager(int predictor_dimension);

    ScalarManagedModel* CreateModel(
      const ScalarStateSpaceSpecification *specification,
      ModelOptions *options,
      std::shared_ptr<PythonListIoManager> io_manager) override;

    StateSpaceRegressionModel * CreateObservationModel(const ScalarStateSpaceSpecification *specification,
      std::shared_ptr<PythonListIoManager> io_manager) override;

    HoldoutErrorSampler CreateHoldoutSampler(
      const ScalarStateSpaceSpecification *specification,
      ModelOptions *options,
      const Vector& responses,
      const Matrix& predictors,
      const std::vector<bool> &response_is_observed,
      int cutpoint,
      int niter,
      bool standardize,
      Matrix *prediction_error_output) override;

  private:  
    void SetSsvsRegressionSampler(const ScalarStateSpaceSpecification *specification, const StateSpaceRegressionModel *sampling_model);
    void SetOdaRegressionSampler(const ScalarStateSpaceSpecification *specification, const StateSpaceRegressionModel *sampling_model);

    int predictor_dimension_;
};

}  // namespace pybsts
}  // namespace BOOM

#endif  // PYBSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_
