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
#include "list_io.hpp"
#include "distributions/rng.hpp"

namespace BOOM {
  namespace pybsts {

    class ScalarModelManager;
    class StateSpaceSpecification;
    class ScalarStateSpaceSpecification;
    class ModelOptions;

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
        void operator()() { impl_->sample_holdout_prediction_errors(); }
      private:
        std::unique_ptr<HoldoutErrorSamplerImpl> impl_;
    };

    class ManagedModel {
      public:
        explicit ManagedModel(ModelOptions *options, std::shared_ptr<PythonListIoManager> io_manager);
        virtual ~ManagedModel() {}

        virtual bool fit(const Vector &y,
                 const Matrix &x = Matrix(),
                 const std::vector<bool> &response_is_observed = std::vector<bool>(),
                 const std::vector<int> &timestamp_indices = std::vector<int>()) = 0;
        virtual Matrix predict(const Matrix &x, const std::vector<int> &forecast_timestamps = std::vector<int>()) = 0;
        virtual Vector SimulateForecast() = 0;

        const int NumberOfTimePoints() const { return number_of_time_points_; }

        int TimestampMapping(int i) const {
          return timestamps_are_trivial_ ? i : timestamp_mapping_[i] - 1;
        }

        std::shared_ptr<ModelOptions> options() { return options_; }
        std::shared_ptr<PythonListIoManager> io_manager() { return io_manager_; }

        const Vector &final_state() const { return final_state_; }
        const std::vector<int> & ForecastTimestamps() { return forecast_timestamps_; }
        RNG & rng() {return rng_;}
        void seed_internal_rng(int seed) { rng_.seed(seed); }
        void seed_internal_rng() { rng_.seed(); }

        void SetDynamicRegressionStateComponentPositions(
            const std::vector<int> &positions) {
          dynamic_regression_state_positions_ = positions;
        }

      protected:
        Vector final_state_;
        std::shared_ptr<PythonListIoManager> io_manager_;
        virtual void update_forecast_predictors(const Matrix &x, const std::vector<int> &forecast_timestamps) {
          forecast_timestamps_ = forecast_timestamps;
        }
        virtual void update_timestamp_mapping(const std::vector<int> &timestamp_mapping) {
          timestamps_are_trivial_ = false;
          timestamp_mapping_ = timestamp_mapping;
          number_of_time_points_ = timestamp_mapping.size();
        }
        void update_number_of_trivial_timepoints(int size) {
          timestamps_are_trivial_ = true;
          number_of_time_points_ = size;
        }

      private:
        std::shared_ptr<ModelOptions> options_;

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

    class ScalarManagedModel : public ManagedModel {
      public:
        explicit ScalarManagedModel(
            const ScalarStateSpaceSpecification *specification,
            ModelOptions* options,
            ScalarStateSpaceModelBase* sampling_model,
            std::shared_ptr<PythonListIoManager> io_manager);

        bool fit(const Vector &y,
                 const Matrix &x = Matrix(),
                 const std::vector<bool> &response_is_observed = std::vector<bool>(),
                 const std::vector<int> &timestamp_indices = std::vector<int>()) override;
        Matrix predict(const Matrix &x, const std::vector<int> &forecast_timestamps = std::vector<int>()) override;
        virtual void AddData(
          const Vector &response,
          const Matrix &predictors,
          const std::vector<bool> &response_is_observed) = 0;
        virtual void AddData(const Vector &response, const std::vector<bool> &response_is_observed) = 0;


        ScalarStateSpaceModelBase* sampling_model() const { return sampling_model_.get(); }
        virtual void sample_posterior() = 0;
        std::shared_ptr<StateSpaceModelBase> sampling_model_sharedptr() const { return sampling_model_; }

      private:
        std::shared_ptr<ScalarStateSpaceModelBase> sampling_model_;
    };

    class ModelManager {
      public:
        ModelManager();
        virtual ~ModelManager() {}

      protected:

      private:


    };

    //===========================================================================
    class ScalarModelManager : public ModelManager {
      public:
        static ScalarModelManager* Create(const std::string &family_name, const int xdim);
        virtual ScalarManagedModel* CreateModel(
            const ScalarStateSpaceSpecification *specification,
            ModelOptions *options,
            std::shared_ptr<PythonListIoManager> io_manager) = 0;

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
        virtual ScalarStateSpaceModelBase * CreateObservationModel(const ScalarStateSpaceSpecification *specification,
            std::shared_ptr<PythonListIoManager> io_manager) = 0;

        virtual HoldoutErrorSampler CreateHoldoutSampler(
          const ScalarStateSpaceSpecification *specification,
          ModelOptions *options,
          const Vector& response,
          const Matrix& inputdata,
          const std::vector<bool> &response_is_observed,
          int cutpoint,
          int niter,
          bool standardize,
          Matrix *prediction_error_output) = 0;
    };

    class StateSpaceSpecification {
      public:
        StateSpaceSpecification();
        virtual ~StateSpaceSpecification() {}
    };

    class SeasonSpecification {
    public:
      explicit SeasonSpecification(int number_of_seasons=1, int duration=1);
      ~SeasonSpecification() {}

      int number_of_seasons() const { return number_of_seasons_; }
      int duration() const { return duration_; }
    private:
      int number_of_seasons_;
      int duration_;
    };

    class LocalTrendSpecification {
    public:
      explicit LocalTrendSpecification();
      explicit LocalTrendSpecification(
        std::unique_ptr<PriorSpecification> trend_prior,
        std::unique_ptr<PriorSpecification> slope_prior,
        std::unique_ptr<PriorSpecification> slope_bias_prior,
        std::unique_ptr<DoublePrior> trend_df_prior,
        std::unique_ptr<DoublePrior> slope_df_prior,
        std::unique_ptr<PriorSpecification> slope_ar1_prior,
        bool static_intercept=false, bool student_errors=false);
      ~LocalTrendSpecification() {}

      std::shared_ptr<PriorSpecification> trend_prior() { return trend_prior_; }
      std::shared_ptr<PriorSpecification> slope_prior() { return slope_prior_; }
      std::shared_ptr<PriorSpecification> slope_bias_prior() { return slope_bias_prior_; }
      std::shared_ptr<DoublePrior> trend_df_prior() { return trend_df_prior_; }
      std::shared_ptr<DoublePrior> slope_df_prior() { return slope_df_prior_; }
      std::shared_ptr<PriorSpecification> slope_ar1_prior() { return slope_ar1_prior_; }
      bool static_intercept() const { return static_intercept_; }
      bool student_errors() const { return student_errors_; }

    private:
      bool static_intercept_;
      bool student_errors_;
      std::shared_ptr<PriorSpecification> trend_prior_;
      std::shared_ptr<PriorSpecification> slope_prior_;
      std::shared_ptr<PriorSpecification> slope_bias_prior_;
      std::shared_ptr<DoublePrior> trend_df_prior_;
      std::shared_ptr<DoublePrior> slope_df_prior_;
      std::shared_ptr<PriorSpecification> slope_ar1_prior_;
    };

    class HierarchicalModelSpecification {
      public:
        HierarchicalModelSpecification(
          std::unique_ptr<DoublePrior> sigma_mean_prior,
          std::unique_ptr<DoublePrior> shrinkage_prior);
        ~HierarchicalModelSpecification() {}

        std::shared_ptr<DoublePrior> sigma_mean_prior() { return sigma_mean_prior_; }
        std::shared_ptr<DoublePrior> shrinkage_prior() { return shrinkage_prior_; }
      private:
        std::shared_ptr<DoublePrior> sigma_mean_prior_;
        std::shared_ptr<DoublePrior> shrinkage_prior_;
    };
    
    class OdaOptions {
      public:
        explicit OdaOptions(double eigenvalue_fudge_factor=0.01, double fallback_probability=0.0);
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
          std::unique_ptr<PriorSpecification> initial_state_prior,
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<LocalTrendSpecification> local_trend);
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> initial_state_prior,
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrendSpecification> local_trend,
          const std::vector<std::string> &predictor_names = std::vector<std::string>(),
          int ar_order = 0, bool dynamic_regression=false);
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> initial_state_prior,
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrendSpecification> local_trend,
          std::unique_ptr<PriorSpecification> ar_prior,
          const std::vector<std::string> &predictor_names = std::vector<std::string>(),
          bool dynamic_regression=false);
        explicit ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> initial_state_prior,
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> seasonal_sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrendSpecification> local_trend,
          const std::string &bma_method, std::unique_ptr<OdaOptions> oda_options,
          const std::vector<SeasonSpecification> &seasons,
          std::unique_ptr<HierarchicalModelSpecification> hierarchical_regression_specification,
          std::unique_ptr<PriorSpecification> ar_prior=nullptr,
          const std::vector<std::string> &predictor_names = std::vector<std::string>(), int ar_order = 0,
          bool dynamic_regression=false);

        std::shared_ptr<PriorSpecification> initial_state_prior() const { return initial_state_prior_; }
        std::shared_ptr<PriorSpecification> sigma_prior() const { return sigma_prior_; }
        std::shared_ptr<PriorSpecification> seasonal_sigma_prior() const { return seasonal_sigma_prior_; }
        std::shared_ptr<PriorSpecification> predictors_prior() const { return predictors_prior_; }
        std::shared_ptr<OdaOptions> oda_options() const { return oda_options_; }
        std::shared_ptr<LocalTrendSpecification> local_trend() const { return local_trend_; }
        const std::string bma_method() const { return bma_method_; }
        int ar_order() const { return ar_order_; }
        std::shared_ptr<PriorSpecification> ar_prior() const { return ar_prior_; }
        bool dynamic_regression() const { return dynamic_regression_; }
        std::shared_ptr<HierarchicalModelSpecification> hierarchical_regression_specification() const { return hierarchical_regression_specification_; }
        std::vector<SeasonSpecification> seasons() const { return seasons_; }
        std::vector<Date> holidays() const { return holidays_; }
        std::vector<std::string> predictor_names() const { return predictor_names_; }

      private:
        std::shared_ptr<PriorSpecification> sigma_prior_;
        std::shared_ptr<PriorSpecification> seasonal_sigma_prior_;
        std::shared_ptr<PriorSpecification> initial_state_prior_;
        std::shared_ptr<PriorSpecification> predictors_prior_;
        int ar_order_;
        std::shared_ptr<PriorSpecification> ar_prior_;
        bool dynamic_regression_;
        std::shared_ptr<HierarchicalModelSpecification> hierarchical_regression_specification_;
        std::string bma_method_;
        std::shared_ptr<OdaOptions> oda_options_;
        std::vector<SeasonSpecification> seasons_;
        std::vector<bool> inclusion_probabilities_;
        std::vector<Date> holidays_;
        std::shared_ptr<LocalTrendSpecification> local_trend_;
        std::vector<std::string> predictor_names_;
    };

    class ModelOptions {
      public:
        explicit ModelOptions(bool save_state_contributions=false, bool save_full_state=false,
          bool save_prediction_errors=false, int niter=100, int ping=10, int burn=0, int forecast_horizon = 1,
          double timeout_threshold_seconds=5.0);
        ~ModelOptions() {}
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


    //======================================================================
    // A callback class for computing the contribution of each state model
    // (including a regression component if there is one) at each time
    // point.
    class ScalarStateContributionCallback
        : public MatrixIoCallback {
     public:
      explicit ScalarStateContributionCallback(ScalarStateSpaceModelBase *model)
          : model_(model),
            has_regression_(-1) {}

      int nrow() const override {
        return model_->number_of_state_models() + has_regression();
      }
      int ncol() const override {return model_->time_dimension();}
      BOOM::Matrix get_matrix() const override {
        BOOM::Matrix ans(nrow(), ncol());
        for (int state = 0; state < model_->number_of_state_models(); ++state) {
          ans.row(state) = model_->state_contribution(state);
        }
        if (has_regression()) {
          ans.last_row() = model_->regression_contribution();
        }
        return ans;
      }

      bool has_regression() const {
        if (has_regression_ == -1) {
          Vector regression_contribution = model_->regression_contribution();
          has_regression_ = !regression_contribution.empty();
        }
        return has_regression_;
      }

     private:
      ScalarStateSpaceModelBase *model_;
      mutable int has_regression_;
    };

    //======================================================================
    // A callback class for saving one step ahead prediction errors from
    // the Kalman filter.
    class PredictionErrorCallback : public VectorIoCallback {
     public:
      explicit PredictionErrorCallback(ScalarStateSpaceModelBase *model)
          : model_(model) {}

      // Each element is a vector of one step ahead prediction errors, so
      // the dimension is the time dimension of the model.
      int dim() const override {
        return model_->time_dimension();
      }

      Vector get_vector() const override {
        return model_->one_step_prediction_errors();
      }

     private:
      ScalarStateSpaceModelBase *model_;
    };

    // A callback class for saving log likelihood values.
    class LogLikelihoodCallback : public ScalarIoCallback {
     public:
      explicit LogLikelihoodCallback(ScalarStateSpaceModelBase *model)
          : model_(model) {}
      double get_value() const override {
        return model_->log_likelihood();
      }
     private:
      ScalarStateSpaceModelBase *model_;
    };

    class FullStateCallback : public MatrixIoCallback {
     public:
      explicit FullStateCallback(StateSpaceModelBase *model) : model_(model) {}
      int nrow() const override {return model_->state_dimension();}
      int ncol() const override {return model_->time_dimension();}
      Matrix get_matrix() const override {return model_->state();}
     private:
      StateSpaceModelBase *model_;
    };

    void seed_global_rng(int seed);
    void seed_global_rng();
  }  // namespace pybsts
}  // namespace BOOM

#endif  // PYBSTS_SRC_MODEL_MANAGER_H_
