#include "prior_specification.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnBase.hpp"
#include "Models/MvnModel.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  namespace pybsts {

    SpikeSlabGlmPrior::SpikeSlabGlmPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips)
        : prior_inclusion_probabilities_(prior_inclusion_probabilities),
          spike_(new VariableSelectionPrior(prior_inclusion_probabilities_)),
          slab_(new MvnModel(prior_mean, prior_precision, true)),
          max_flips_(max_flips)
    { }

    RegressionConjugateSpikeSlabPrior::RegressionConjugateSpikeSlabPrior(
        const Vector &prior_inclusion_probabilities,
        const Vector &prior_mean, const SpdMatrix &prior_precision, 
        const Vector &prior_variance_diagonal, int max_flips,
        double prior_df, double sigma_guess, double sigma_upper_limit,
        const Ptr<UnivParams> &residual_variance)
        : prior_inclusion_probabilities_(prior_inclusion_probabilities),
          spike_(new VariableSelectionPrior(prior_inclusion_probabilities_)),
          siginv_prior_(new ChisqModel(prior_df, sigma_guess)),
          max_flips_(max_flips),
          sigma_upper_limit_(sigma_upper_limit)
    {
      if (prior_variance_diagonal.size() > 0) {
        slab_.reset(new IndependentMvnModelGivenScalarSigma(prior_mean, prior_variance_diagonal, residual_variance));
      } else {
        slab_.reset(new MvnGivenScalarSigma(prior_mean, prior_precision, residual_variance));
      }
    }

    namespace {
      typedef StudentRegressionConjugateSpikeSlabPrior SRCSSP;
      typedef StudentRegressionNonconjugateSpikeSlabPrior SRNSSP;
      typedef StudentIndependentSpikeSlabPrior SISSP;
    }

    SRCSSP::StudentRegressionConjugateSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
        const Vector &prior_mean, const SpdMatrix &prior_precision, 
        const Vector &prior_variance_diagonal, int max_flips,
        double prior_df, double sigma_guess, double sigma_upper_limit,
        Ptr<DoubleModel> df_prior_model, const Ptr<UnivParams> &residual_variance
        )
        : RegressionConjugateSpikeSlabPrior(prior_inclusion_probabilities,
            prior_mean, prior_precision, prior_variance_diagonal, max_flips, 
            prior_df, sigma_guess, sigma_upper_limit, residual_variance),
          df_prior_(df_prior_model)
    {}

    RegressionNonconjugateSpikeSlabPrior::RegressionNonconjugateSpikeSlabPrior(
        const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips,
          double prior_df, double sigma_guess, double sigma_upper_limit)
        : SpikeSlabGlmPrior(prior_inclusion_probabilities, prior_mean, prior_precision, max_flips),
          sigma_upper_limit_(sigma_upper_limit),
          siginv_prior_(new ChisqModel(prior_df, sigma_guess))
    {}

    ArSpikeSlabPrior::ArSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips, 
          double prior_df, double sigma_guess, double sigma_upper_limit, 
          bool truncate)
        : RegressionNonconjugateSpikeSlabPrior(prior_inclusion_probabilities, prior_mean,
            prior_precision, max_flips, prior_df, sigma_guess, sigma_upper_limit),
          truncate_(truncate)
    {}

    SRNSSP::StudentRegressionNonconjugateSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips, 
          double prior_df, double sigma_guess, double sigma_upper_limit,
          Ptr<DoubleModel> df_prior_model)
        : RegressionNonconjugateSpikeSlabPrior(prior_inclusion_probabilities, prior_mean,
            prior_precision, max_flips, prior_df, sigma_guess, sigma_upper_limit),
          df_prior_(df_prior_model)
    {}

    IndependentRegressionSpikeSlabPrior::IndependentRegressionSpikeSlabPrior(
          const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const Vector &prior_variance_diagonal,
          int max_flips, double prior_df, double sigma_guess, double sigma_upper_limit,
          const Ptr<UnivParams> &sigsq)
        : prior_inclusion_probabilities_(prior_inclusion_probabilities),
          spike_(new VariableSelectionPrior(prior_inclusion_probabilities_)),
          slab_(new IndependentMvnModelGivenScalarSigma(
              prior_mean, prior_variance_diagonal, sigsq)),
          siginv_prior_(new ChisqModel(prior_df, sigma_guess)),
          max_flips_(max_flips),
          sigma_upper_limit_(sigma_upper_limit)
    {}

    SISSP::StudentIndependentSpikeSlabPrior(
        const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const Vector &prior_variance_diagonal,
          int max_flips, double prior_df, double sigma_guess, double sigma_upper_limit,
          Ptr<DoubleModel> df_prior_model, const Ptr<UnivParams> &sigsq)
        : IndependentRegressionSpikeSlabPrior(prior_inclusion_probabilities, 
            prior_mean, prior_variance_diagonal,
            max_flips, prior_df, sigma_guess, sigma_upper_limit,
            sigsq),
          df_prior_(df_prior_model)
    {}

  }  // namespace pybsts
}  // namespace BOOM
