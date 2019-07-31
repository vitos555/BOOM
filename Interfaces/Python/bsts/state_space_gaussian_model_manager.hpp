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

#ifndef PYBSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_
#define PYBSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_

#include "model_manager.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"

namespace BOOM {
namespace pybsts {

// A base class that handles "CreateModel" for both the regression and
// non-regression flavors of Gaussian models.
class GaussianModelManagerBase : public ScalarModelManager {
 protected:
  virtual void init_io_manager(
            ScalarStateSpaceModelBase* sampling_model,
            ModelOptions *options,
            std::shared_ptr<PythonListIoManager> io_manager);
};

class StateSpaceManagedModel : public ScalarManagedModel {
  public:
    StateSpaceManagedModel(const ScalarStateSpaceSpecification *specification,
            ModelOptions* options,
            ScalarStateSpaceModelBase* sampling_model,
            std::shared_ptr<PythonListIoManager> io_manager);
    void AddData(const Vector &response, const std::vector<bool> &response_is_observed) override;
    void AddData(
          const Vector &response,
          const Matrix &predictors,
          const std::vector<bool> &response_is_observed) override;
    void sample_posterior() override;
    Vector SimulateForecast() override;
  private:
};

// A holdout error sampler for a plain Gaussian state space model.
class StateSpaceModelPredictionErrorSampler
    : public HoldoutErrorSamplerImpl {
 public:
  // Args:
  //   model:  The model containing data up to a specified cutpoint.
  //   holdout_data:  Observed values after the cutpoint.
  //   niter: The desired number of draws (MCMC iterations) from the posterior
  //     distribution.
  //   errors:  A matrix that will hold the output of the simulation.
  StateSpaceModelPredictionErrorSampler(std::unique_ptr<StateSpaceManagedModel> model,
                                        const Vector &holdout_data,
                                        int niter,
                                        bool standardize,
                                        Matrix *errors);
  void sample_holdout_prediction_errors() override;

 private:
  std::unique_ptr<StateSpaceManagedModel> model_;
  Vector holdout_data_;
  int niter_;
  bool standardize_;
  Matrix *errors_;
};

class StateSpaceModelManager
    : public GaussianModelManagerBase {
 public:
  ScalarManagedModel* CreateModel(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    std::shared_ptr<PythonListIoManager> io_manager) override;
  StateSpaceModel * CreateObservationModel(const ScalarStateSpaceSpecification *specification,
    std::shared_ptr<PythonListIoManager> io_manager) override;

  HoldoutErrorSampler CreateHoldoutSampler(
    const ScalarStateSpaceSpecification *specification,
    ModelOptions *options,
    const Vector& response,
    const Matrix& inputdata,
    const std::vector<bool> &response_is_observed,
    int cutpoint,
    int niter,
    bool standardize,
    Matrix *prediction_error_output) override;

 private:

};

}  // namespace pybsts
}  // namespace BOOM

#endif  // PYBSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_
