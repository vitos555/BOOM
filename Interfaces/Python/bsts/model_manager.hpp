// Copyright 2019 UMG. All Rights Reserved.
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

#ifndef PYBSTS_SRC_MODEL_MANAGER_H_
#define PYBSTS_SRC_MODEL_MANAGER_H_

#include <string>
#include <vector>
#include <memory>

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "cpputil/Date.hpp"
#include "prior_specification.hpp"

namespace BOOM {
  namespace pybsts {

    class ScalarModelManager;
    class StateSpaceSpecification;
    class ScalarStateSpaceSpecification;
    class PyBstsOptions;

    //===========================================================================
    // The code that computes out of sample one-step prediction errors is
    // designed for multi-threading.  This base class provides the interface for
    // computing the prediction errors.
    class HoldoutErrorSamplerImpl {
      public:
        virtual ~HoldoutErrorSamplerImpl() {}

        // Simulate from the distribution of one-step prediction errors given data
        // up to some cutpoint.  Child classes must be equipped with enough state
        // to carry out this operation and store the results in an appropriate
        // data structure.
        virtual void sample_holdout_prediction_errors() = 0;
    };

    // A null class that can be used by model families that do not support one
    // step prediction errors (e.g. logit and Poisson).
    class NullErrorSampler : public HoldoutErrorSamplerImpl {
      public:
        void sample_holdout_prediction_errors() override {}
    };

    // A pimpl-based functor for computing out of sample prediction errors, with
    // the appropriate interface for submitting to a ThreadPool.
    class HoldoutErrorSampler {
      public:
        explicit HoldoutErrorSampler(HoldoutErrorSamplerImpl *impl)
          : impl_(impl) {}
        void operator()() {impl_->sample_holdout_prediction_errors();}
      private:
        std::unique_ptr<HoldoutErrorSamplerImpl> impl_;
    };

    class ModelManager {
      public:
        ModelManager();
        virtual ~ModelManager() {}

        virtual bool fit(Matrix x, Vector y) = 0;
        virtual Matrix predict(Matrix x) = 0;

        const int NumberOfTimePoints() const { return number_of_time_points_; }

        int TimestampMapping(int i) const {
          return timestamps_are_trivial_ ? i : timestamp_mapping_[i] - 1;
        }

        const std::vector<int> & ForecastTimestamps() { return forecast_timestamps_; }

        std::shared_ptr<PyBstsOptions> options() const { return options_; }
        std::shared_ptr<ScalarStateSpaceModelBase> observation_model() const { return observation_model_; }

        RNG & rng() {return rng_;}     
      protected:
        std::shared_ptr<ScalarStateSpaceModelBase> observation_model_;
        std::shared_ptr<PyBstsOptions> options_;
        Vector final_state_;

      private:
        RNG rng_;

        //----------------------------------------------------------------------
        // Time stamps are trivial the timestamp information was NULL, or if there
        // is at most one observation at each time point.
        bool timestamps_are_trivial_;

        // The number of distinct time points.
        int number_of_time_points_;

        // Indicates the time point (in R's 1-based counting system) to which each
        // observation belongs.
        std::vector<int> timestamp_mapping_;

        // Indicates the number of time points past the end of the training data
        // for each forecast data point.  If data are not multiplexed then
        // forecast_timestamps_ will be empty.
        std::vector<int> forecast_timestamps_;

        // The index of each dynamic regression state component is stored here,
        // where 'index' refers to the state component's position in the list of
        // state models stored in the primary state space model.
        std::vector<int> dynamic_regression_state_positions_;

    };

    //===========================================================================
    class ScalarModelManager : public ModelManager {
      public:
        static ScalarModelManager* Create(const std::string &family_name, const int xdim);
        ScalarStateSpaceModelBase* CreateModel(const ScalarStateSpaceSpecification *specification, const PyBstsOptions *options);
        bool InitializeModel(const ScalarStateSpaceSpecification *specification, const PyBstsOptions *options);

        bool fit(Matrix x, Vector y) override;
        Matrix predict(Matrix x) override;

      private:
        // Create the specific StateSpaceModel suitable for the given model
        // family.  The posterior sampler for the model is set, and entries for
        // its model parameters are created in io_manager.  This function does not
        // add state to the the model.  It is primarily intended to aid the
        // implementation of CreateModel.
        //
        // The arguments are documented in the comment to CreateModel.
        //
        // Returns:
        //   A pointer to the created model.  The pointer is owned by a Ptr in the
        //   the child class, so working with the raw pointer privately is
        //   exception safe.
        virtual ScalarStateSpaceModelBase * CreateObservationModel(const ScalarStateSpaceSpecification *specification) = 0;

        virtual HoldoutErrorSampler CreateHoldoutSampler(
          const ScalarStateSpaceSpecification *specification,
          const PyBstsOptions *options,
          const Vector& response,
          const Matrix& inputdata,
          const std::vector<bool> &response_is_observed,
          int cutpoint,
          int niter,
          bool standardize,
          Matrix *prediction_error_output) = 0;

        // This function must not be called before UnpackForecastData.  It takes
        // the current state of the model held by the child classes, along with
        // the data obtained by UnpackForecastData(), and simulates one draw from
        // the posterior predictive forecast distribution.
       virtual Vector SimulateForecast(const Vector &final_state) = 0;
    };

    class StateSpaceSpecification {
      public:
        StateSpaceSpecification();
        virtual ~StateSpaceSpecification() {}
    };

    class Season {
    public:
      explicit Season(int duration);
      ~Season() {}

    private:
      int duration_;
    };

    class LocalTrend {
    public:
      explicit LocalTrend(bool has_intercept=false, bool has_trend=false,
        bool has_slope=false, bool slope_has_bias=false, bool student_errors=false);
      ~LocalTrend() {}

      bool has_intercept() const { return has_intercept_; }
      bool has_trend() const { return has_trend_; }
      bool has_slope() const { return has_slope_; }
      bool slope_has_bias() const { return slope_has_bias_; }
      bool student_errors() const { return student_errors_; }

    private:
      bool has_intercept_;
      bool has_trend_;
      bool has_slope_;
      bool slope_has_bias_;
      bool student_errors_;
    };
    
    class PriorSpecification {
      public:
        explicit PriorSpecification();
        ~PriorSpecification() {}

        const Vector &prior_inclusion_probabilities() const { return prior_inclusion_probabilities_; }
        const Vector &prior_mean() const { return prior_mean_; }
        const SpdMatrix &prior_precision() const { return prior_precision_; }
        const Vector &prior_variance_diagonal() const { return prior_variance_diagonal_; }
        int max_flips() const { return max_flips_; }
        double prior_df() const { return prior_df_; }
        double sigma_guess() const { return sigma_guess_; }
        double sigma_upper_limit() const { return sigma_upper_limit_; }

      private:
        Vector prior_inclusion_probabilities_;
        Vector prior_mean_;
        SpdMatrix prior_precision_;
        Vector prior_variance_diagonal_;
        int max_flips_;
        double prior_df_;
        double sigma_guess_;
        double sigma_upper_limit_;
    };

    class OdaOptions {
      public:
        explicit OdaOptions(double eigenvalue_fudge_factor, double fallback_probability);
        ~OdaOptions() {}

        double eigenvalue_fudge_factor() const { return eigenvalue_fudge_factor_; }
        double fallback_probability() const { return fallback_probability_; }

      private:
        double eigenvalue_fudge_factor_;
        double fallback_probability_;

    };

    class ScalarStateSpaceSpecification : public StateSpaceSpecification {
      public:
        explicit ScalarStateSpaceSpecification();
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<LocalTrend> local_trend);
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrend> local_trend,
          int ar_order = 0);
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrend> local_trend,
          const std::string &bma_method, std::unique_ptr<OdaOptions> oda_options,
          int ar_order = 0);

        std::shared_ptr<PriorSpecification> sigma_prior() const { return sigma_prior_; }
        std::shared_ptr<PriorSpecification> predictors_prior() const { return predictors_prior_; }
        std::shared_ptr<OdaOptions> oda_options() const { return oda_options_; }
        std::shared_ptr<LocalTrend> local_trend() const { return local_trend_; }
        const std::string bma_method() const { return bma_method_; }

      private:
        std::shared_ptr<PriorSpecification> sigma_prior_;
        std::shared_ptr<PriorSpecification> initial_state_prior_;
        std::shared_ptr<PriorSpecification> predictors_prior_;
        int ar_order_;
        std::string bma_method_;
        std::shared_ptr<OdaOptions> oda_options_;
        std::vector<Season> seasons_;
        std::vector<bool> inclusion_probabilities_;
        std::vector<Date> holidays_;
        std::shared_ptr<LocalTrend> local_trend_;
    };

    class PyBstsOptions {
      public:
        explicit PyBstsOptions(bool save_state_contributions=false, bool save_full_state=false,
          bool save_prediction_errors=false, int niter=100, int ping=1, double timeout_threshold_seconds=5.0);
        ~PyBstsOptions() {}
        const bool save_state_contributions() const { return save_state_contributions_; }
        const bool save_prediction_errors() const { return save_prediction_errors_; }
        const bool save_full_state() const { return save_full_state_; }
        int niter() const { return niter_; }
        int ping() const { return ping_; }
        int burn() const { return burn_; }
        int forecast_horizon() const { return forecast_horizon_; }
        double timeout_threshold_seconds() const { return timeout_threshold_seconds_; }

      private:
        bool save_state_contributions_;
        bool save_prediction_errors_;
        bool save_full_state_;
        int niter_;
        int ping_;
        int burn_;
        int forecast_horizon_;
        double timeout_threshold_seconds_;
    };

  }  // namespace pybsts
}  // namespace BOOM

#endif  // PYBSTS_SRC_MODEL_MANAGER_H_
