/*
  Copyright (C) 2005-2015 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "Models/StateSpace/StateSpaceLogitModel.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/Constants.hpp"

namespace BOOM {
  namespace {
    typedef StateSpaceLogitModel SSLM;
    typedef StateSpace::AugmentedBinomialRegressionData ABRD;
  }  // namespace

  ABRD::AugmentedBinomialRegressionData()
      : state_model_offset_(0.0)
  {}

  ABRD::AugmentedBinomialRegressionData(
      double y, double n, const Vector &x)
      : state_model_offset_(0.0)
  {
    add_data(new BinomialRegressionData(y, n, x));
  }

  ABRD::AugmentedBinomialRegressionData(
      const std::vector<Ptr<BinomialRegressionData>> &binomial_data)
      : state_model_offset_(0.0)
  {
    for (int i = 0; i < binomial_data.size(); ++i) {
      add_data(binomial_data[i]);
    }
  }

  ABRD * ABRD::clone() const {
    return new ABRD(*this);
  }

  std::ostream & ABRD::display(std::ostream &out) const {
    out << "state model offset:  " << state_model_offset_ << std::endl
        << std::setw(10) << "y"
        << std::setw(10) << "n"
        << std::setw(12) << "latent value"
        << std::setw(10) << "precision "
        << "predictors" << std::endl;
    for (int i = 0; i < binomial_data_.size(); ++i) {
      out << std::setw(10) << binomial_data_[i]->y()
          << std::setw(10) << binomial_data_[i]->n()
          << std::setw(12) << latent_continuous_values_[i]
          << std::setw(10) << precisions_[i]
          << binomial_data_[i]->x()
          << std::endl;
    }
    return out;
  }

  // Initial values for latent data are arbitrary, but must be legal
  // values.
  void ABRD::add_data(const Ptr<BinomialRegressionData> &binomial_data) {
    MultiplexedData::add_data(binomial_data);
    binomial_data_.push_back(binomial_data);
    latent_continuous_values_.push_back(0);
    precisions_.push_back(binomial_data->missing() == Data::observed ?
                          4.0 / binomial_data->n() : 0);
  }

  void ABRD::set_latent_data(double value, double precision, int observation) {
    if (precision < 0) {
      report_error("precision must be non-negative.");
    }
    precisions_[observation] = precision;
    latent_continuous_values_[observation] = value;
  }

  double ABRD::latent_data_variance(int observation) const {
    return 1.0 / precisions_[observation];
  }

  double ABRD::latent_data_value(int observation) const {
    return latent_continuous_values_[observation];
  }

  double ABRD::adjusted_observation(const GlmCoefs &coefficients) const {
    if (missing() == Data::completely_missing
        || binomial_data_.empty()) {
      return negative_infinity();
    }
    double total_precision = 0;
    double ans = 0;
    for (int i = 0; i < binomial_data_.size(); ++i) {
      if (binomial_data(i).missing() == Data::observed) {
        ans += precisions_[i] *
            (latent_continuous_values_[i] -
             coefficients.predict(binomial_data_[i]->x()));
        total_precision += precisions_[i];
      }
    }
    if (total_precision <= 0 || !std::isfinite(total_precision)) {
      return negative_infinity();
    }
    return ans / total_precision;
  }

  double ABRD::latent_data_overall_variance() const {
    if (missing() == Data::observed && observed_sample_size() > 0) {
      // This is the normal case, where all observations at a time point are
      // observed.
        return 1.0 / sum(precisions_);
    } else if (missing() == Data::completely_missing
               || observed_sample_size() == 0) {
      // In the case of NO information about a time point, the observation
      // variance is the variance of the standard logistic distribution, which
      // is pi^2 / 3.
        return Constants::pi_squared / 3.0;
    } else {
      // If neither of the preceding cases holds, then there is partial
      // information, we just have to be careful to only include the observed
      // cases.
      double total_precision = 0.0;
      for (int i = 0; i < binomial_data_.size(); ++i) {
        if (binomial_data_[i]->missing() == Data::observed) {
          total_precision += precisions_[i];
        }
      }
      return 1.0 / total_precision;
    }
  }

  void ABRD::set_state_model_offset(double offset) {
    state_model_offset_ = offset;
  }

  double ABRD::total_trials() const {
    double ans = 0;
    for (int i = 0; i < binomial_data_.size(); ++i) {
      ans += binomial_data_[i]->n();
    }
    return ans;
  }

  double ABRD::total_successes() const {
    double ans = 0;
    for (int i = 0; i < binomial_data_.size(); ++i) {
      ans += binomial_data_[i]->y();
    }
    return ans;
  }

  //======================================================================
  void SSLM::setup() {
    observe(observation_model_->coef_prm());
  }

  SSLM::StateSpaceLogitModel(int xdim)
      : StateSpaceNormalMixture(xdim > 1),
        observation_model_(new BinomialLogitModel(xdim))
  {
    setup();
  }

  SSLM::StateSpaceLogitModel(const Vector &successes,
                             const Vector &trials,
                             const Matrix &design,
                             const std::vector<bool> &observed)
      : StateSpaceNormalMixture(ncol(design)),
        observation_model_(new BinomialLogitModel(ncol(design)))
  {
    setup();
    bool all_observed = observed.empty();
    if (successes.size() != trials.size()
        || successes.size() != nrow(design)
        || (!all_observed && successes.size() != observed.size())) {
      report_error("Data sizes do not match in StateSpaceLogitModel "
                   "constructor");
    }
    for (int i = 0; i < successes.size(); ++i) {
      NEW(ABRD, dp)(successes[i],
                    trials[i],
                    design.row(i));
      if (!(all_observed || observed[i])) {
        dp->set_missing_status(Data::missing_status::completely_missing);
        dp->binomial_data_ptr(0)->set_missing_status(
            Data::missing_status::completely_missing);
      }
      add_data(dp);
    }
  }

  SSLM::StateSpaceLogitModel(const SSLM &rhs)
      : StateSpaceNormalMixture(rhs),
        observation_model_(rhs.observation_model_->clone())
  {}

  SSLM * SSLM::clone() const {return new SSLM(*this);}

  int SSLM::time_dimension() const {
    return dat().size();
  }

  double SSLM::observation_variance(int t) const {
    if (t > time_dimension()) {
      return Constants::pi_squared / 3.0;
    }
    return dat()[t]->latent_data_overall_variance();
  }

  double SSLM::adjusted_observation(int t) const {
    if (is_missing_observation(t)) {
      return negative_infinity();
    }
    return dat()[t]->adjusted_observation(observation_model_->coef());
  }

  bool SSLM::is_missing_observation(int t) const {
    return t > time_dimension()
        || dat()[t]->missing() == Data::completely_missing
        || dat()[t]->observed_sample_size() == 0;
  }

  void SSLM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      dat()[t]->set_state_model_offset(observation_matrix(t).dot(state(t)));
      signal_complete_data_change(t);
    }
  }

  Vector SSLM::simulate_forecast(RNG &rng,
                                 const Matrix &forecast_predictors,
                                 const Vector &trials,
                                 const Vector &final_state) {
    StateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(nrow(forecast_predictors));
    Vector state = final_state;
    int t0 = dat().size();
    for (int t = 0; t < ans.size(); ++t) {
      state = simulate_next_state(rng, state, t + t0);
      double eta = observation_matrix(t + t0).dot(state)
          + observation_model_->predict(forecast_predictors.row(t));
      double probability = plogis(eta);
      ans[t] = rbinom_mt(rng, lround(trials[t]), probability);
    }
    return ans;
  }

  Vector StateSpaceLogitModel::one_step_holdout_prediction_errors(
      RNG &rng,
      BinomialLogitDataImputer &data_imputer,
      const Vector &successes,
      const Vector &trials,
      const Matrix &predictors,
      const Vector &final_state) {
    if (nrow(predictors) != successes.size()
        || trials.size() != successes.size()) {
      report_error("Size mismatch in arguments provided to "
                   "one_step_holdout_prediction_errors.");
    }
    Vector ans(successes.size());
    int t0 = dat().size();
    ScalarKalmanStorage ks(state_dimension());
    ks.a = *state_transition_matrix(t0 - 1) * final_state;
    ks.P = SpdMatrix(state_variance_matrix(t0 - 1)->dense());

    // This function differs from the Gaussian case because the
    // response is on the binomial scale, and the state model is on
    // the logit scale.  Because of the nonlinearity, we need to
    // incorporate the uncertainty about the forecast in the
    // prediction for the observation.  We do this by imputing the
    // latent logit and its mixture indicator for each observation.
    // The strategy is (for each observation)
    //   1) simulate next state.
    //   2) simulate w_t given state
    //   3) kalman update state given w_t.
    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      Vector state = rmvn(ks.a, ks.P);

      double state_contribution = observation_matrix(t+t0).dot(state);
      double regression_contribution =
          observation_model_->predict(predictors.row(t));
      double mu = state_contribution + regression_contribution;
      double prediction = trials[t] * plogis(mu);
      ans[t] = successes[t] - prediction;

      // ans[t] is a random draw of the one step ahead prediction
      // error at time t0+t given observed data to time t0+t-1.  We
      // now proceed with the steps needed to update the Kalman filter
      // so we can compute ans[t+1].

      double precision_weighted_sum, total_precision;
      std::tie(precision_weighted_sum, total_precision) = data_imputer.impute(
          rng,
          trials[t],
          successes[t],
          mu);
      double latent_observation = precision_weighted_sum / total_precision;
      double latent_variance = 1.0 / total_precision;

      // The latent state was drawn from its predictive distribution
      // given Y[t0 + t -1] and used to impute the latent data for
      // y[t0+t].  That latent data is now used to update the Kalman
      // filter for the next time period.  It is important that we
      // discard the imputed state at this point.
      sparse_scalar_kalman_update(
          latent_observation - regression_contribution,
          ks.a,
          ks.P,
          ks.K,
          ks.F,
          ks.v,
          missing,
          observation_matrix(t + t0),
          latent_variance,
          *state_transition_matrix(t + t0),
          *state_variance_matrix(t + t0));
    }
    return ans;
  }

}  // namespace BOOM
