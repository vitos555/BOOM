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

#include "Models/StateSpace/Filters/KalmanTools.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"

#include "distributions.hpp"

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

    // The model manager will be thread safe as long as it is created from the
    // home thread.
    ModelManager::ModelManager()
        : rng_(seed_rng(GlobalRng::rng)),
          timestamps_are_trivial_(true),
          number_of_time_points_(-1) {}

    // ScalarModelManager * ScalarModelManager::Create(std::vector<std::string> py_bsts_object) {  
    //   std::string family = ToString(getListElement(py_bsts_object, "family"));
    //   bool regression = !Py_isNull(getListElement(py_bsts_object, "predictors"));
    //   int xdim = 0;
    //   if (regression) {
    //     xdim = Py_ncols(getListElement(py_bsts_object, "predictors"));
    //   }
    //   return ScalarModelManager::Create(family, xdim);
    // }

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
            << " in ModelManager::Create.";
        report_error(err.str());
      }
      return nullptr;
    }

    ScalarStateSpaceModelBase* ScalarModelManager::CreateModel(
        const ScalarStateSpaceSpecification *specification, 
        const PyBstsOptions *options) {
      ScalarStateSpaceModelBase* model = nullptr;
      if (specification) {
        model = CreateObservationModel(specification);
      }
      if (options) {
        // Initialize state contributions vector
        if (options->save_state_contributions()) {
        }

        // Initialize prediction errors vector
        if (options->save_prediction_errors()) {
        }

        // Initialize full state vector
        if (options->save_full_state()) {
        }
      }
      return model;
    }    

    bool ScalarModelManager::InitializeModel(const ScalarStateSpaceSpecification *specification, 
        const PyBstsOptions *options) {
      bool ret = false;
      if (specification) {
        observation_model_.reset(CreateObservationModel(specification));
        ret = true;
      }
      if (options) {
        options_ = std::make_shared<PyBstsOptions>(options);

        // Initialize state contributions vector
        if (options_->save_state_contributions()) {
        }

        // Initialize prediction errors vector
        if (options_->save_prediction_errors()) {
        }

        // Initialize full state vector
        if (options_->save_full_state()) {
        }
      } else {
        ret = false;
      }
      return ret;
    }

    bool ScalarModelManager::fit(Matrix x, Vector y) {
      try {

        // Do one posterior sampling step before getting ready to write.  This
        // will ensure that any dynamically allocated objects have the correct
        // size
        observation_model_->sample_posterior();
        int niter = options_->niter();
        int ping = options_->ping();
        double timeout_threshold_seconds = options_->timeout_threshold_seconds();

        clock_t start_time = clock();
        double time_threshold = CLOCKS_PER_SEC * timeout_threshold_seconds;
        for (int i = 0; i < niter; ++i) {
          BOOM::print_python_timestamp(i, ping);
          try {
            observation_model_->sample_posterior();
            // io_manager.write();
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
              std::cout << warning.str();
              // return BOOM::appendListElement(
              //     ans,
              //     ToRVector(BOOM::Vector(1, i + 1)),
              //     "ngood");
            }
          } catch(std::exception &e) {
            std::ostringstream err;
            err << "Caught an exception with the following "
                << "error message in MCMC "
                << "iteration " << i << ".  Aborting." << std::endl
                << e.what() << std::endl;
          }
        }
      } catch (std::exception &e) {
        std::cerr << "Got exception: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "Got unknown exception" << std::endl;
      }
      return true;
    }

    // Primary implementation of predict.bsts.  Child classes will carry out
    // some of the details, but most of the prediction logic is here.
    Matrix ScalarModelManager::predict(Matrix x) {
      Vector final_state = observation_model_->final_state();
      int niter = options_->niter();
      int burn = options_->burn();
      int forecast_horizon = options_->forecast_horizon();
      int iterations_after_burnin = niter - burn;
      int refilter = false;

      if (x.ncol() == 0) {
        report_error("Forecast called with 0 columns of prediction data.");
      }

      Matrix ans(iterations_after_burnin, forecast_horizon);
      for (int i = 0; i < iterations_after_burnin; ++i) {
        // io_manager.stream();
        if (refilter) {
          observation_model_->kalman_filter();
          const Kalman::ScalarMarginalDistribution &marg(
              observation_model_->get_filter().back());
          Vector state_mean = marg.state_mean();
          SpdMatrix state_variance = marg.state_variance();
          make_contemporaneous(
              state_mean,
              state_variance,
              marg.prediction_variance(),
              marg.prediction_error(),
              observation_model_->observation_matrix(observation_model_->time_dimension()).dense());
          final_state = rmvn(state_mean, state_variance);
        }
        ans.row(i) = SimulateForecast(final_state);
      }
      return ans;
    }

    // void ModelManager::UnpackDynamicRegressionForecastData(
    //     StateSpaceModelBase *model,
    //     std::vector<std::string> py_state_specification,
    //     std::vector<std::string> py_prediction_data) {
    //   if (Py_length(py_state_specification) < model->number_of_state_models()) {
    //     std::ostringstream err;
    //     err << "The number of state components in the model: ("
    //         << model->number_of_state_models() << ") does not match the size of "
    //         << "the state specification: ("
    //         << Py_length(py_state_specification)
    //         << ") in UnpackDynamicRegressionForecastData.";
    //     report_error(err.str());
    //   }
    //   std::deque<int> positions(dynamic_regression_state_positions().begin(),
    //                             dynamic_regression_state_positions().end());
    //   for (int i = 0; i < model->number_of_state_models(); ++i) {
    //     std::vector<std::string> spec = VECTOR_ELT(py_state_specification, i);
    //     if (Py_inherits(spec, "DynamicRegression")) {
    //       Matrix predictors = ToBoomMatrix(getListElement(
    //           py_prediction_data, "dynamic.regression.predictors"));
    //       if (positions.empty()) {
    //         report_error("Found a previously unseen dynamic regression state "
    //                      "component.");
    //       }
    //       int pos = positions[0];
    //       positions.pop_front();
    //       Ptr<StateModel> state_model = model->state_model(pos);
    //       state_model.dcast<DynamicRegressionStateModel>()->add_forecast_data(
    //           predictors);
    //     }
    //   }
    // }

    // void ModelManager::UnpackTimestampInfo(std::vector<std::string> py_data_list) {
    //   std::vector<std::string> py_timestamp_info = getListElement(py_data_list, "timestamp.info");
    //   timestamps_are_trivial_ = Py_asLogical(getListElement(
    //       py_timestamp_info, "timestamps.are.trivial"));
    //   number_of_time_points_ = Py_asInteger(getListElement(
    //       py_timestamp_info, "number.of.time.points"));
    //   if (!timestamps_are_trivial_) {
    //     timestamp_mapping_ = ToIntVector(getListElement(
    //         py_timestamp_info, "timestamp.mapping"));
    //   }
    // }

    // void ModelManager::UnpackForecastTimestamps(std::vector<std::string> py_prediction_data) {
    //   std::vector<std::string> py_forecast_timestamps = getListElement(
    //       py_prediction_data, "timestamps");
    //   if (!Py_isNull(py_forecast_timestamps)) {
    //     forecast_timestamps_ = ToIntVector(getListElement(
    //         py_forecast_timestamps, "timestamp.mapping"));
    //     for (int i = 1; i < forecast_timestamps_.size(); ++i) {
    //       if (forecast_timestamps_[i] < forecast_timestamps_[i - 1]) {
    //         report_error("Time stamps for multiplex predictions must be "
    //                      "in increasing order.");
    //       }
    //     }
    //   }
    // }

    Season::Season(int duration) : duration_(duration)
    {}

    PyBstsOptions::PyBstsOptions(bool save_state_contributions, bool save_full_state,
      bool save_prediction_errors, int niter, int ping, double timeout_threshold_seconds) : 
      save_state_contributions_(save_state_contributions), save_full_state_(save_full_state),
      save_prediction_errors_(save_prediction_errors), niter_(niter), ping_(ping),
      timeout_threshold_seconds_(timeout_threshold_seconds)
    {}

    LocalTrend::LocalTrend(bool has_intercept, bool has_trend, bool has_slope, 
        bool slope_has_bias, bool student_errors) : student_errors_(student_errors) {
      if (has_intercept && (has_trend || has_slope || slope_has_bias)) {
        std::ostringstream err;
        err << "Incopatible local trend: can not have local intercept ond any other local trend";
        report_error(err.str());
      } else {
        has_intercept_ = has_intercept;
        has_trend_ = has_trend;
        has_slope_ = has_slope;
        slope_has_bias_ = slope_has_bias;
      }
    }

    OdaOptions::OdaOptions(double eigenvalue_fudge_factor, double fallback_probability) : eigenvalue_fudge_factor_(eigenvalue_fudge_factor), fallback_probability_(fallback_probability) 
    {}

    StateSpaceSpecification::StateSpaceSpecification() {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification() : ar_order_(0), bma_method_("ODA")
    {
      local_trend_ = std::make_shared<LocalTrend>(new LocalTrend());
    }

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<LocalTrend> local_trend
      ) : 
      sigma_prior_(std::move(sigma_prior)), local_trend_(std::move(local_trend)), ar_order_(0), bma_method_("ODA")
    {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<PriorSpecification> predictors_prior,
      std::unique_ptr<LocalTrend> local_trend,
      int ar_order) : 
      sigma_prior_(std::move(sigma_prior)), predictors_prior_(std::move(predictors_prior)),
      ar_order_(ar_order), bma_method_("ODA"),
      local_trend_(std::move(local_trend))
    {}

    ScalarStateSpaceSpecification::ScalarStateSpaceSpecification(
      std::unique_ptr<PriorSpecification> sigma_prior,
      std::unique_ptr<PriorSpecification> predictors_prior,
      std::unique_ptr<LocalTrend> local_trend,
      const std::string &bma_method, std::unique_ptr<OdaOptions> oda_options,
      int ar_order) : 
      sigma_prior_(std::move(sigma_prior)), predictors_prior_(std::move(predictors_prior)),
      ar_order_(ar_order), bma_method_(bma_method),
      oda_options_(std::move(oda_options)), local_trend_(std::move(local_trend))
    {}


  }  // namespace pybsts
}  // namespace BOOM
