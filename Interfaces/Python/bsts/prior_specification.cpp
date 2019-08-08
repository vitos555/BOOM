/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include<math.h>

#include "prior_specification.hpp"
#include "Models/BetaModel.hpp"
#include "Models/DiscreteUniformModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/TruncatedGammaModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/LognormalModel.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/PoissonModel.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/UniformModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace pybsts {
    DoublePrior::DoublePrior(const std::string &family, 
      double a, double b, double a_truncation, double b_truncation) :
      family_(family), a_(a), b_(b), a_truncation_(a_truncation), b_truncation_(b_truncation)
    {}

    PriorSpecification::PriorSpecification(
          Vector prior_inclusion_probabilities,
          Vector prior_mean,
          SpdMatrix prior_precision,
          Vector prior_variance_diagonal,
          int max_flips,
          double initial_value,
          double mu,
          double prior_df,
          double prior_guess,
          double sigma_guess,
          double sigma_upper_limit,
          bool truncate,
          bool positive,
          bool fixed
          ) : 
      prior_inclusion_probabilities_(prior_inclusion_probabilities),
      prior_mean_(prior_mean),
      prior_precision_(prior_precision),
      prior_variance_diagonal_(prior_variance_diagonal),
      max_flips_(max_flips),
      initial_value_(initial_value),
      mu_(mu),
      prior_df_(prior_df),
      prior_guess_(prior_guess),
      sigma_guess_(sigma_guess),
      sigma_upper_limit_(sigma_upper_limit),
      truncate_(truncate),
      positive_(positive),
      fixed_(fixed)
    {}

    SdPrior::SdPrior(double prior_guess, double prior_df, double initial_value, bool fixed, double upper_limit)
        : prior_guess_(prior_guess),
          prior_df_(prior_df),
          initial_value_(initial_value),
          fixed_(fixed),
          upper_limit_(upper_limit)
    {
      if (upper_limit_ < 0 || !std::isfinite(upper_limit_)) {
        upper_limit_ = BOOM::infinity();
      }
    }

    std::ostream & SdPrior::print(std::ostream &out) const {
      out << "prior_guess_   = " << prior_guess_ << std::endl
          << "prior_df_      = " << prior_df_ << std::endl
          << "initial_value_ = " << initial_value_ << std::endl
          << "fixed          = " << fixed_ << std::endl
          << "upper_limit_   = " << upper_limit_ << std::endl;
      return out;
    }

    NormalPrior::NormalPrior(double mu, double sigma, double initial_value, bool fixed)
        : mu_(mu),
          sigma_(sigma),
          initial_value_(initial_value),
          fixed_(fixed) 
    {}

    std::ostream & NormalPrior::print(std::ostream &out) const {
      out << "mu =     " << mu_ << std::endl
          << "sigma_ = " << sigma_ << std::endl
          << "init   = " << initial_value_ << std::endl;
      return out;
    }

    BetaPrior::BetaPrior(double a, double b, double initial_value)
        : a_(a),
          b_(b),
          initial_value_(initial_value)
    {}

    std::ostream & BetaPrior::print(std::ostream &out) const {
      out << "a = " << a_ << "b = " << b_;
      return out;
    }

    GammaPrior::GammaPrior(double a, double b, double initial_value)
        : a_(a),
          b_(b)
    {
      if (initial_value) {
        initial_value_ = initial_value;
      } else {
        initial_value_ = a_ / b_;
      }
    }

    std::ostream & GammaPrior::print(std::ostream &out) const {
      out << "a = " << a_ << "b = " << b_;
      return out;
    }

    TruncatedGammaPrior::TruncatedGammaPrior(double a, double b, double initial_value, double lower_truncation_point, double upper_truncation_point)
        : GammaPrior(a, b, initial_value),
          lower_truncation_point_(lower_truncation_point),
          upper_truncation_point_(upper_truncation_point)
    {}

    std::ostream & TruncatedGammaPrior::print(std::ostream &out) const {
      GammaPrior::print(out) << " (" << lower_truncation_point_
          << ", " << upper_truncation_point_ << ") ";
      return out;
    }

    MvnPrior::MvnPrior(const Vector &mu, const SpdMatrix &Sigma)
        : mu_(mu),
          Sigma_(Sigma)
    {}

    std::ostream & MvnPrior::print(std::ostream &out) const {
      out << "mu: " << mu_ << std::endl
          << "Sigma:" << std::endl
          << Sigma_;
      return out;
    }

    Ar1CoefficientPrior::Ar1CoefficientPrior(double mu, double sigma, double initial_value, bool fixed, bool force_stationary, bool force_positive)
        : NormalPrior(mu, sigma, initial_value, fixed),
          force_stationary_(force_stationary),
          force_positive_(force_positive) {}

    std::ostream & Ar1CoefficientPrior::print(std::ostream &out) const {
      NormalPrior::print(out) << "force_stationary_ = "
                              << force_stationary_ << std::endl;
      return out;
    }

    ConditionalNormalPrior::ConditionalNormalPrior(double mu, double sample_size)
        : mu_(mu),
          sample_size_(sample_size)
    {}

    std::ostream & ConditionalNormalPrior::print(std::ostream & out) const {
      out << "prior mean: " << mu_ << std::endl
          << "prior sample size for prior mean:" << sample_size_;
      return out;
    }

    NormalInverseGammaPrior::NormalInverseGammaPrior(double prior_mean_guess, double prior_mean_sample_size, const SdPrior &sd_prior)
        : prior_mean_guess_(prior_mean_guess),
          prior_mean_sample_size_(prior_mean_sample_size),
          sd_prior_(sd_prior)
    {}

    std::ostream & NormalInverseGammaPrior::print(std::ostream &out) const {
      out << "prior_mean_guess        = " << prior_mean_guess_ << std::endl
          << "prior_mean_sample_size: = " << prior_mean_sample_size_
          << std::endl
          << "prior for sigma: " << std::endl
          << sd_prior_;
      return out;
    }

    DirichletPrior::DirichletPrior(const Vector &prior_counts)
        : prior_counts_(prior_counts)
    {}

    const Vector & DirichletPrior::prior_counts() const {
      return prior_counts_;
    }

    int DirichletPrior::dim() const {
      return prior_counts_.size();
    }

    MarkovPrior::MarkovPrior(const Matrix &transition_counts, const Vector &initial_state_counts)
        : transition_counts_(transition_counts),
          initial_state_counts_(initial_state_counts)
    {}

    std::ostream & MarkovPrior::print(std::ostream &out) const {
      out << "prior transition counts: " << std::endl
          << transition_counts_ << std::endl
          << "prior initial state counts:" << std::endl
          << initial_state_counts_;
      return out;
    }

    MarkovModel * MarkovPrior::create_markov_model() const {
      MarkovModel * ans(new MarkovModel(transition_counts_.nrow()));
      Ptr<MarkovConjSampler> sampler(new MarkovConjSampler(
          ans, transition_counts_, initial_state_counts_));
      ans->set_method(sampler);
      return ans;
    }

    InverseWishartPrior::InverseWishartPrior(double variance_guess_weight, const SpdMatrix &variance_guess)
        : variance_guess_weight_(variance_guess_weight),
          variance_guess_(variance_guess)
    {}

    NormalInverseWishartPrior::NormalInverseWishartPrior(const Vector &mu_guess, double mu_guess_weight, const SpdMatrix &sigma_guess, double sigma_guess_weight)
        : mu_guess_(mu_guess),
          mu_guess_weight_(mu_guess_weight),
          sigma_guess_(sigma_guess),
          sigma_guess_weight_(sigma_guess_weight)
    {}

    std::ostream & NormalInverseWishartPrior::print(std::ostream &out) const {
      out << "the prior mean for mu:" << std::endl
          << mu_guess_ << std::endl
          << "prior sample size for mu0: " << mu_guess_weight_ << std::endl
          << "prior sample size for Sigma_guess: " << sigma_guess_weight_
          << std::endl
          << "prior guess at Sigma: " << std::endl
          << sigma_guess_ << std::endl;
      return out;
    }

    MvnIndependentSigmaPrior::MvnIndependentSigmaPrior(const MvnPrior &mu_prior, std::vector<SdPrior> sigma_priors)
        : mu_prior_(mu_prior)
    {
      int n = mu_prior_.mu().size();
      sigma_priors_.reserve(n);
      for (auto el = sigma_priors.begin(); el != sigma_priors.end(); el++) {
        SdPrior sigma_prior = *el;
        sigma_priors_.push_back(sigma_prior);
      }
    }

    MvnDiagonalPrior::MvnDiagonalPrior(const Vector &mean, const Vector &sd)
        : mean_(mean),
          sd_(sd)
    {}

    DiscreteUniformPrior::DiscreteUniformPrior(int lo, int hi)
        :lo_(lo),
         hi_(hi)
    {
      if (hi_ < lo_) {
        report_error("hi < lo in DiscreteUniformPrior.");
      }
      log_normalizing_constant_ = -log1p(hi_ - lo_);
    }

    double DiscreteUniformPrior::logp(int value) const {
      if (value < lo_ || value > hi_) {
        return negative_infinity();
      }
      return log_normalizing_constant_;
    }

    PoissonPrior::PoissonPrior(double lambda, double lo, double hi)
        : lambda_(lambda),
          lo_(lo),
          hi_(hi)
    {
      if (lambda_ <= 0) {
        report_error("lambda must be positive in PoissonPrior");
      }
      if (hi_ < lo_) {
        report_error("upper.limit < lower.limit in PoissonPrior.");
      }
      log_normalizing_constant_ = log(ppois(hi_, lambda_)
                                      - ppois(lo_ - 1, lambda_));
    }

    double PoissonPrior::logp(int value) const {
      return dpois(value, lambda_, true) - log_normalizing_constant_;
    }

    PointMassPrior::PointMassPrior(int location)
        : location_(location)
    {}

    double PointMassPrior::logp(int value) const {
      return value == location_ ? 0 : negative_infinity();
    }

    RegressionCoefficientConjugatePrior::RegressionCoefficientConjugatePrior(
        const Vector &mean, double sample_size, const Vector &additional_prior_precision, double diagonal_weight)
        : mean_(mean),
          sample_size_(sample_size),
          additional_prior_precision_(additional_prior_precision),
          diagonal_weight_(diagonal_weight)
    {}

    UniformPrior::UniformPrior(double lo, double hi, double initial_value)
        : lo_(lo),
          hi_(hi),
          initial_value_(initial_value)
    {}

    // Ptr<LocationScaleDoubleModel> create_location_scale_double_model(
    //     SEXP r_spec, bool throw_on_failure) {
    //   if (Rf_inherits(r_spec, "GammaPrior")) {
    //     GammaPrior spec(r_spec);
    //     return new GammaModel(spec.a(), spec.b());
    //   } else if (Rf_inherits(r_spec, "BetaPrior")) {
    //     BetaPrior spec(r_spec);
    //     return new BetaModel(spec.a(), spec.b());
    //   } else if (Rf_inherits(r_spec, "NormalPrior")) {
    //     NormalPrior spec(r_spec);
    //     return new GaussianModel(spec.mu(), spec.sigma() * spec.sigma());
    //   } else if (Rf_inherits(r_spec, "UniformPrior")) {
    //     double lo = Rf_asReal(getListElement(r_spec, "lo"));
    //     double hi = Rf_asReal(getListElement(r_spec, "hi"));
    //     return new UniformModel(lo, hi);
    //   } else if (Rf_inherits(r_spec, "LognormalPrior")) {
    //     double mu = Rf_asReal(getListElement(r_spec, "mu"));
    //     double sigma = Rf_asReal(getListElement(r_spec, "sigma"));
    //     return new LognormalModel(mu, sigma);
    //   }
    //   if (throw_on_failure) {
    //     report_error("Could not convert specification into a "
    //                  "LocationScaleDoubleModel");
    //   }
    //   return nullptr;
    // }

    // Ptr<DoubleModel> create_double_model(SEXP r_spec) {
    //   Ptr<LocationScaleDoubleModel> ans =
    //       create_location_scale_double_model(r_spec, false);
    //   if (!!ans) {
    //     return ans;
    //   } else if (Rf_inherits(r_spec, "TruncatedGammaPrior")) {
    //     TruncatedGammaPrior spec(r_spec);
    //     return new TruncatedGammaModel(
    //         spec.a(), spec.b(), spec.lower_truncation_point(),
    //         spec.upper_truncation_point());
    //   }
    //   report_error("Could not convert specification into a DoubleModel");
    //   return nullptr;
    // }

    // Ptr<DiffDoubleModel> create_diff_double_model(SEXP r_spec) {
    //   if (Rf_inherits(r_spec, "GammaPrior")) {
    //     GammaPrior spec(r_spec);
    //     return new GammaModel(spec.a(), spec.b());
    //   } else if (Rf_inherits(r_spec, "TruncatedGammaPrior")) {
    //     TruncatedGammaPrior spec(r_spec);
    //     return new TruncatedGammaModel(
    //         spec.a(), spec.b(), spec.lower_truncation_point(),
    //         spec.upper_truncation_point());
    //   } else if (Rf_inherits(r_spec, "BetaPrior")) {
    //     BetaPrior spec(r_spec);
    //     return new BetaModel(spec.a(), spec.b());
    //   } else if (Rf_inherits(r_spec, "NormalPrior")) {
    //     NormalPrior spec(r_spec);
    //     return new GaussianModel(spec.mu(), spec.sigma() * spec.sigma());
    //   } else if (Rf_inherits(r_spec, "SdPrior")) {
    //     SdPrior spec(r_spec);
    //     double shape = spec.prior_df() / 2;
    //     double sum_of_squares = square(spec.prior_guess()) * spec.prior_df();
    //     double scale = sum_of_squares / 2;
    //     if (spec.upper_limit() < infinity()) {
    //       double lower_limit = 1.0 / square(spec.upper_limit());
    //       double upper_limit = infinity();
    //       return new TruncatedGammaModel(shape, scale, lower_limit,
    //                                      upper_limit);
    //     } else {
    //       return new GammaModel(shape, scale);
    //     }
    //   } else if (Rf_inherits(r_spec, "UniformPrior")) {
    //     UniformPrior spec(r_spec);
    //     return new UniformModel(spec.lo(), spec.hi());
    //   }
    //   report_error("Could not convert specification into a DiffDoubleModel");
    //   return nullptr;
    // }

    // Ptr<IntModel> create_int_model(SEXP r_spec) {
    //   if (Rf_inherits(r_spec, "DiscreteUniformPrior")) {
    //     DiscreteUniformPrior spec(r_spec);
    //     return new DiscreteUniformModel(spec.lo(), spec.hi());
    //   } else if (Rf_inherits(r_spec, "PoissonPrior")) {
    //     PoissonPrior spec(r_spec);
    //     return new PoissonModel(spec.lambda());
    //   } else if (Rf_inherits(r_spec, "PointMassPrior")) {
    //     PointMassPrior spec(r_spec);
    //     return new DiscreteUniformModel(spec.location(), spec.location());
    //   } else {
    //     report_error("Could not convert specification into an IntModel.");
    //     return nullptr;
    //   }
    // }

    Ptr<LocationScaleDoubleModel> create_double_model(
        std::shared_ptr<DoublePrior> spec) {
      if (spec->family() == "GammaPrior") {
        return new GammaModel(spec->a(), spec->b());
      } else if (spec->family() == "BetaPrior") {
        return new BetaModel(spec->a(), spec->b());
      } else if (spec->family() == "NormalPrior") {
        return new GaussianModel(spec->a(), spec->b() * spec->b());
      } else if (spec->family() == "UniformPrior") {
        return new UniformModel(spec->a(), spec->b());
      } else if (spec->family() == "LognormalPrior") {
        return new LognormalModel(spec->a(), spec->b());
      } else if (spec->family() == "TruncatedGammaPrior") {
        return new TruncatedGammaModel(
            spec->a(), spec->b(), spec->a_truncation(),
            spec->b_truncation());
      }
      return nullptr;
    }

  }  // namespace pybsts
}  // namespace BOOM
