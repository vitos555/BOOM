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

#include <string>
#include <time.h>
#include <cstdio>

#include "model_manager.hpp"
#include "state_space_gaussian_model_manager.hpp"
#include "state_space_regression_model_manager.hpp"
#include "create_state_model.hpp"

#include "Models/StateSpace/Filters/KalmanTools.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"

#include "distributions.hpp"
#include "distributions/rng.hpp"

namespace BOOM {
  void print_python_timestamp(int iteration_number, int ping){
    if (ping <= 0) return;
    if ( (iteration_number % ping) == 0) {
      time_t rawtime;
      time(&rawtime);
#ifdef _WIN32
      // mingw does not include the re-entrant versions localtime_r
      // and asctime_r.
      std::string time_str(asctime(localtime(&rawtime)));
#else
      struct tm timeinfo;
      localtime_r(&rawtime, &timeinfo);
      char buf[28];
      std::string time_str(asctime_r(&timeinfo, buf));
      time_str = time_str.substr(0, time_str.find("\n"));
#endif
      const char *sep="=-=-=-=-=";
      std::printf("%s Iteration %d %s %s\n", sep, iteration_number,
              time_str.c_str(), sep);
    }
  }

  namespace pybsts {

    ManagedModel::ManagedModel(ModelOptions *options,
        std::shared_ptr<PythonListIoManager> io_manager) :
      rng_(seed_rng(GlobalRng::rng)),
      timestamps_are_trivial_(true),
      number_of_time_points_(-1),
      io_manager_(io_manager)
    {
      if (options) {
        options_.reset(options);
      } else {
        report_error("Can't create model without options.");
      }      
    }

    // The model manager will be thread safe as long as it is created from the
    // home thread.
    ModelManager::ModelManager()
    {}


    ScalarModelManager *ScalarModelManager::Create(
        const std::string &family_name, int xdim) {
      if (family_name == "gaussian") {
        if (xdim > 0) {
          StateSpaceRegressionModelManager *manager =
              new StateSpaceRegressionModelManager(xdim);
          return manager;
        } else {
          return new StateSpaceModelManager;
        }
      } else {
        std::ostringstream err;
        err << "Unrecognized family name: " << family_name
            << " in ModelManager::Create." << endl;
        report_error(err.str());
      }
      return nullptr;
    }

    ScalarManagedModel::ScalarManagedModel(
            const ScalarStateSpaceSpecification *specification,
            ModelOptions *input_options,
            ScalarStateSpaceModelBase* input_sampling_model,
            std::shared_ptr<PythonListIoManager> input_io_manager) :
      ManagedModel(input_options, input_io_manager)
    {
      if (input_sampling_model) {
        sampling_model_.reset(input_sampling_model);
        StateModelFactory state_model_factory(io_manager());
        state_model_factory.AddState(this, specification, "");
        SetDynamicRegressionStateComponentPositions(
            state_model_factory.DynamicRegressionStateModelPositions());
        state_model_factory.SaveFinalState(sampling_model(), &final_state_);
        // Initialize state contributions vector
        if ((options()) && (io_manager())) {
          if (options()->save_state_contributions()) {
            io_manager()->add_list_element(
                new NativeMatrixListElement(
                    new ScalarStateContributionCallback(sampling_model()),
                    "state.contributions",
                    nullptr));
          }

          // Initialize prediction errors vector
          if (options()->save_prediction_errors()) {
            io_manager()->add_list_element(
                new BOOM::NativeVectorListElement(
                    new PredictionErrorCallback(sampling_model()),
                    "one.step.prediction.errors",
                    nullptr));          
          }

          // Initialize full state vector
          if (options()->save_full_state()) {
            io_manager()->add_list_element(
                new NativeMatrixListElement(
                    new FullStateCallback(sampling_model()), "full.state", nullptr));
          }
        }
      } else {
        report_error("Empty sampling_model in ScalarManagedModel::ScalarManagedModel.");
      }
    }

    bool ScalarManagedModel::fit(const Vector &y,
            const Matrix &x,
            const std::vector<bool> &response_is_observed,
            const std::vector<int> &timestamp_indices) {
      try {
        // Do one posterior sampling step before getting ready to write.  This
        // will ensure that any dynamically allocated objects have the correct
        // size
        std::vector<bool> response_is_observed_;
        if (!response_is_observed.empty()) {
          response_is_observed_ = response_is_observed;
        } else {
          response_is_observed_ = std::vector<bool>(y.size(), true);
        }
        if (timestamp_indices.empty()) {
          update_number_of_trivial_timepoints(y.size());
        } else {
          update_timestamp_mapping(timestamp_indices);
        }
        if (x.ncol() == 0) {
          AddData(y, response_is_observed_);
        } else {
          AddData(y, x, response_is_observed_);
        }
        sample_posterior();
        int niter = options()->niter();
        int ping = options()->ping();
        double timeout_threshold_seconds = options()->timeout_threshold_seconds();
        io_manager()->prepare_to_write(niter);

        clock_t start_time = clock();
        double time_threshold = CLOCKS_PER_SEC * timeout_threshold_seconds;
        for (int i = 0; i < niter; ++i) {
          BOOM::print_python_timestamp(i, ping);
          try {
            sample_posterior();
            io_manager()->write();
            clock_t current_time = clock();
            if (current_time - start_time > time_threshold) {
              std::ostringstream warning;
              warning << "Timeout threshold "
                      << time_threshold
                      << " exceeded in iteration " << i << "."
                      << std::endl
                      << "Time used was "
                      << double(current_time - start_time) / CLOCKS_PER_SEC
                      << " seconds.";
              report_warning(warning.str());
            }
          } catch(std::exception &e) {
            std::ostringstream err;
            err << "Caught an exception with the following "
                << "error message in MCMC "
                << "iteration " << i << ".  Aborting." << std::endl
                << e.what() << std::endl;
            report_error(err.str());
          }
        }
      } catch (std::exception &e) {
        std::ostringstream err;
        err << "Got exception: " << e.what() << std::endl;
        report_error(err.str());
      } catch (...) {
        std::ostringstream err;
        err << "Got unknown exception" << std::endl;
        report_error(err.str());
      }
      return true;
    }

    // Primary implementation of predict.  Child classes will carry out
    // some of the details, but most of the prediction logic is here.
    Matrix ScalarManagedModel::predict(const Matrix &x, const std::vector<int> &forecast_timestamps) {
      int niter = options()->niter();
      int burn = options()->burn();
      int forecast_horizon = options()->forecast_horizon();
      int iterations_after_burnin = niter - burn;
      bool refilter = false;

      if ((x.nrow() > 0) && (x.nrow() != forecast_timestamps.size())) {
        report_error("Missmatch between number of rows in predictors and forecast indices.");
      }

      if (x.nrow() > 0) {
        forecast_horizon = x.nrow();
        update_forecast_predictors(x, forecast_timestamps);
      }
      io_manager()->prepare_to_stream();
      if (burn > 0) {
        io_manager()->advance(burn);
      }

      Matrix ans(iterations_after_burnin, forecast_horizon);
      for (int i = 0; i < iterations_after_burnin; ++i) {
        io_manager()->stream();
        if (refilter) {
          sampling_model()->kalman_filter();
          const Kalman::ScalarMarginalDistribution &marg(
              sampling_model()->get_filter().back());
          Vector state_mean = marg.state_mean();
          SpdMatrix state_variance = marg.state_variance();
          make_contemporaneous(
              state_mean,
              state_variance,
              marg.prediction_variance(),
              marg.prediction_error(),
              sampling_model()->observation_matrix(this->sampling_model()->time_dimension()).dense());
          final_state_ = rmvn(state_mean, state_variance);
        }
        ans.row(i) = SimulateForecast();
      }
      return ans;
    }

    SeasonSpecification::SeasonSpecification(int number_of_seasons, int duration) : 
      number_of_seasons_(number_of_seasons),
      duration_(duration)
    {}

    ModelOptions::ModelOptions(bool save_state_contributions, bool save_full_state,
      bool save_prediction_errors, int niter, int ping, int burn, int forecast_horizon,
      double timeout_threshold_seconds) : 
      save_state_contributions_(save_state_contributions), save_full_state_(save_full_state),
      save_prediction_errors_(save_prediction_errors), niter_(niter), ping_(ping), burn_(burn),
      forecast_horizon_(forecast_horizon), timeout_threshold_seconds_(timeout_threshold_seconds)
    {}

    LocalTrendSpecification::LocalTrendSpecification() :
      static_intercept_(false),
      student_errors_(false)
    {}


    LocalTrendSpecification::LocalTrendSpecification(
        std::unique_ptr<PriorSpecification> trend_prior,
        std::unique_ptr<PriorSpecification> slope_prior,
        std::unique_ptr<PriorSpecification> slope_bias_prior,
        std::unique_ptr<DoublePrior> trend_df_prior,
        std::unique_ptr<DoublePrior> slope_df_prior,
        std::unique_ptr<PriorSpecification> slope_ar1_prior,
        bool static_intercept, bool student_errors) : 
      static_intercept_(static_intercept),
      student_errors_(student_errors),
      trend_prior_(std::move(trend_prior)),
      slope_prior_(std::move(slope_prior)),
      slope_bias_prior_(std::move(slope_bias_prior)),
      trend_df_prior_(std::move(trend_df_prior)),
      slope_df_prior_(std::move(slope_df_prior)),
      slope_ar1_prior_(std::move(slope_ar1_prior_))
    {
      if (static_intercept_ && (trend_prior_ || slope_prior_ || slope_bias_prior_)) {
        std::ostringstream warning;
        warning << "Reseting intercept. Local trend can not have static intercept ond any other local trend";
        report_warning(warning.str());
        static_intercept_ = false;
      }
    }

    HierarchicalModelSpecification::HierarchicalModelSpecification(
          std::unique_ptr<DoublePrior> sigma_mean_prior,
          std::unique_ptr<DoublePrior> shrinkage_prior) :
      sigma_mean_prior_(std::move(sigma_mean_prior)),
      shrinkage_prior_(std::move(shrinkage_prior))
    {}

    OdaOptions::OdaOptions(double eigenvalue_fudge_factor, double fallback_probability) : 
      eigenvalue_fudge_factor_(eigenvalue_fudge_factor), fallback_probability_(fallback_probability) 
    {}

    StateSpaceSpecification::StateSpaceSpecification() {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification() : 
      ar_order_(0), bma_method_("ODA"),
      dynamic_regression_(false)
    {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> initial_state_prior,
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<LocalTrendSpecification> local_trend) : 
      initial_state_prior_(std::move(initial_state_prior)),
      sigma_prior_(std::move(sigma_prior)), local_trend_(std::move(local_trend)), ar_order_(0), bma_method_("ODA"),
      dynamic_regression_(false)
    {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> initial_state_prior,
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<PriorSpecification> predictors_prior,
      std::unique_ptr<LocalTrendSpecification> local_trend,
      const std::vector<std::string> &predictor_names,
      int ar_order, bool dynamic_regression) : 
      initial_state_prior_(std::move(initial_state_prior)),
      sigma_prior_(std::move(sigma_prior)), predictors_prior_(std::move(predictors_prior)),
      ar_order_(ar_order), bma_method_("ODA"),
      local_trend_(std::move(local_trend)),
      predictor_names_(predictor_names),
      dynamic_regression_(dynamic_regression)
    {}
    
    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
          std::unique_ptr<PriorSpecification> initial_state_prior,
          std::unique_ptr<PriorSpecification> sigma_prior,
          std::unique_ptr<PriorSpecification> predictors_prior,
          std::unique_ptr<LocalTrendSpecification> local_trend,
          std::unique_ptr<PriorSpecification> ar_prior,
          const std::vector<std::string> &predictor_names,
          bool dynamic_regression) : 
      initial_state_prior_(std::move(initial_state_prior)),
      sigma_prior_(std::move(sigma_prior)), predictors_prior_(std::move(predictors_prior)),
      ar_order_(-1), ar_prior_(std::move(ar_prior)), bma_method_("ODA"),
      local_trend_(std::move(local_trend)),
      predictor_names_(predictor_names),
      dynamic_regression_(dynamic_regression)
    {}


    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> initial_state_prior,
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<PriorSpecification> seasonal_sigma_prior,
      std::unique_ptr<PriorSpecification> predictors_prior,
      std::unique_ptr<LocalTrendSpecification> local_trend,
      const std::string &bma_method, std::unique_ptr<OdaOptions> oda_options,
      const std::vector<SeasonSpecification> &seasons,
      std::unique_ptr<HierarchicalModelSpecification> hierarchical_regression_specification,
      std::unique_ptr<PriorSpecification> ar_prior,
      const std::vector<std::string> &predictor_names,
      int ar_order, bool dynamic_regression) : 
      initial_state_prior_(std::move(initial_state_prior)),
      sigma_prior_(std::move(sigma_prior)),
      seasonal_sigma_prior_(std::move(seasonal_sigma_prior)),
      predictors_prior_(std::move(predictors_prior)),
      ar_order_(ar_order), ar_prior_(std::move(ar_prior)), bma_method_(bma_method),
      oda_options_(std::move(oda_options)), local_trend_(std::move(local_trend)),
      hierarchical_regression_specification_(std::move(hierarchical_regression_specification)),
      predictor_names_(predictor_names),
      seasons_(seasons),
      dynamic_regression_(dynamic_regression)
    {}

    void seed_global_rng(int seed) {
      BOOM::GlobalRng::rng.seed(seed);
      srand(seed);
    }
    void seed_global_rng() {
      BOOM::GlobalRng::rng.seed();
    }

  }  // namespace pybsts
}  // namespace BOOM
