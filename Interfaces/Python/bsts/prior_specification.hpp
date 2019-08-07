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

#ifndef BOOM_PYBSTS_PRIOR_SPECIFICATION_HPP_
#define BOOM_PYBSTS_PRIOR_SPECIFICATION_HPP_

#include "Models/BetaModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/UniformModel.hpp"
#include "Models/LognormalModel.hpp"
#include "Models/TruncatedGammaModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/IndependentMvnModel.hpp"
#include "Models/IndependentMvnModelGivenScalarSigma.hpp"
#include "Models/MvnBase.hpp"
#include "Models/MvnGivenScalarSigma.hpp"

namespace BOOM{

  class MarkovModel;

  namespace pybsts{
    // Convenience classes for communicating commonly used Python objects
    // to BOOM.

    class DoublePrior {
      public:
        DoublePrior(const std::string &family="", double a=0.0, double b=0.0, double a_truncation=0.0, double b_truncation=0.0);
        ~DoublePrior() {}

        const std::string& family() const { return family_; }
        double a() const { return a_; }
        double b() const { return b_; }
        double a_truncation() const { return a_truncation_; }
        double b_truncation() const { return b_truncation_; }

      private:
        std::string family_;
        double a_;
        double b_;
        double a_truncation_;
        double b_truncation_;
    };

    class PriorSpecification {
      public:
        explicit PriorSpecification(
          Vector prior_inclusion_probabilities=Vector(),
          Vector prior_mean=Vector(),
          SpdMatrix prior_precision=SpdMatrix(),
          Vector prior_variance_diagonal=Vector(),
          int max_flips=0,
          double initial_value=0.0,
          double mu=0.0,
          double prior_df=0.0,
          double prior_guess=0.0,
          double sigma_guess=1.0,
          double sigma_upper_limit=1.0,
          bool truncate=false,
          bool positive=false,
          bool fixed=false
          );
        ~PriorSpecification() {}

        const Vector &prior_inclusion_probabilities() const { return prior_inclusion_probabilities_; }
        const Vector &prior_mean() const { return prior_mean_; }
        const SpdMatrix &prior_precision() const { return prior_precision_; }
        const Vector &prior_variance_diagonal() const { return prior_variance_diagonal_; }
        int max_flips() const { return max_flips_; }
        double initial_value() const { return initial_value_; }
        double mu() const { return mu_; }
        double prior_df() const { return prior_df_; }
        double prior_guess() const { return prior_guess_; }
        double sigma_guess() const { return sigma_guess_; }
        double sigma_upper_limit() const { return sigma_upper_limit_; }
        bool truncate() const { return truncate_; }
        bool positive() const { return positive_; }
        bool fixed() const { return fixed_; }

      private:
        Vector prior_inclusion_probabilities_;
        Vector prior_mean_;
        SpdMatrix prior_precision_;
        Vector prior_variance_diagonal_;
        int max_flips_;
        double initial_value_;
        double mu_;
        double prior_df_;
        double prior_guess_;
        double sigma_guess_;
        double sigma_upper_limit_;
        bool truncate_;
        bool positive_;
        bool fixed_;
    };


    // For encoding an inverse Gamma prior on a variance parameter.
    class SdPrior {
     public:
      explicit SdPrior(double prior_guess, double prior_df, double initial_value, bool fixed, double upper_limit);
      double prior_guess()const {return prior_guess_;}
      double prior_df()const {return prior_df_;}
      double initial_value()const {return initial_value_;}
      bool fixed()const {return fixed_;}
      double upper_limit()const {return upper_limit_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double prior_guess_;
      double prior_df_;
      double initial_value_;
      bool fixed_;
      double upper_limit_;
    };
    //----------------------------------------------------------------------

    // For encoding a Gaussian prior on a scalar.
    class NormalPrior {
     public:
      explicit NormalPrior(double mu, double sigma, double initial_value, bool fixed);
      virtual ~NormalPrior() {}
      virtual std::ostream & print(std::ostream &out) const;
      double mu() const {return mu_;}
      double sigma() const {return sigma_;}
      double sigsq() const {return sigma_ * sigma_;}
      double initial_value() const {return initial_value_;}
      bool fixed() const {return fixed_;}

     private:
      double mu_;
      double sigma_;
      double initial_value_;
      bool fixed_;
    };

    //----------------------------------------------------------------------
    // For encoding a prior on an AR1 coefficient.  This is a Gaussian
    // prior, but users have the option of truncating the support to
    // [-1, 1] to enforce stationarity of the AR1 process.
    class Ar1CoefficientPrior : public NormalPrior {
     public:
      explicit Ar1CoefficientPrior(double mu, double sigma, double initial_value, bool fixed, bool force_stationary, bool force_positive);
      bool force_stationary()const {return force_stationary_;}
      bool force_positive()const {return force_positive_;}
      std::ostream & print(std::ostream &out)const override;

     private:
      bool force_stationary_;
      bool force_positive_;
    };

    //----------------------------------------------------------------------
    // For encoding the parameters in a conditionally normal model.
    // Tyically this is the prior on mu in an normal(mu, sigsq), where
    // mu | sigsq ~ N(mu0, sigsq / sample_size).
    class ConditionalNormalPrior {
     public:
      explicit ConditionalNormalPrior(double mu, double sample_size);
      double prior_mean()const{return mu_;}
      double sample_size()const{return sample_size_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double mu_;
      double sample_size_;
    };

    //----------------------------------------------------------------------
    // A NormalInverseGammaPrior is the conjugate prior for the mean
    // and variance in a normal distribution.
    class NormalInverseGammaPrior {
     public:
      explicit NormalInverseGammaPrior(double prior_mean_guess, double prior_mean_sample_size, const SdPrior &sd_prior);
      double prior_mean_guess()const{return prior_mean_guess_;}
      double prior_mean_sample_size()const{return prior_mean_sample_size_;}
      const SdPrior &sd_prior()const{return sd_prior_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double prior_mean_guess_;
      double prior_mean_sample_size_;
      SdPrior sd_prior_;
    };

    //----------------------------------------------------------------------
    // For encoding the parameters of a Dirichlet distribution.  The R
    // constructor that builds 'prior' ensures that prior_counts_ is a
    // positive length vector of positive reals.
    class DirichletPrior {
     public:
      explicit DirichletPrior(const Vector &prior_counts);
      const Vector & prior_counts()const;
      int dim()const;

     private:
      Vector prior_counts_;
    };

    //----------------------------------------------------------------------
    // For encoding a prior on the parameters of a Markov chain.  This
    // is product Dirichlet prior for the rows of the transition
    // probabilities, and an independent Dirichlet on the initial
    // state distribution.
    // TODO(stevescott): add support for fixing the initial
    //   distribution in various ways.
    class MarkovPrior {
     public:
      explicit MarkovPrior(const Matrix &transition_counts, const Vector &initial_state_counts);
      const Matrix & transition_counts()const {return transition_counts_;}
      const Vector & initial_state_counts()const {return initial_state_counts_;}
      int dim()const {return transition_counts_.nrow();}
      std::ostream & print(std::ostream &out)const;
      // Creates a Markov model with this as a prior.
      BOOM::MarkovModel * create_markov_model()const;

     private:
      Matrix transition_counts_;
      Vector initial_state_counts_;
    };

    //----------------------------------------------------------------------
    class BetaPrior {
     public:
      explicit BetaPrior(double a, double b, double initial_value);
      double a()const{return a_;}
      double b()const{return b_;}
      double initial_value() const {return initial_value_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double a_, b_;
      double initial_value_;
    };

    //----------------------------------------------------------------------
    class GammaPrior {
     public:
      explicit GammaPrior(double a, double b, double initial_value);
      virtual ~GammaPrior(){}
      double a()const{return a_;}
      double b()const{return b_;}
      double initial_value()const{return initial_value_;}
      virtual std::ostream & print(std::ostream &out)const;

     private:
      double a_, b_;
      double initial_value_;
    };

    class TruncatedGammaPrior : public GammaPrior {
     public:
      explicit TruncatedGammaPrior(double a, double b, double initial_value, double lower_truncation_point, double upper_truncation_point);
      double lower_truncation_point() const {return lower_truncation_point_;}
      double upper_truncation_point() const {return upper_truncation_point_;}
      std::ostream &print(std::ostream &out) const override;

     private:
      double lower_truncation_point_;
      double upper_truncation_point_;
    };

    class MvnPrior {
     public:
      explicit MvnPrior(const Vector &mu, const SpdMatrix &Sigma);
      const Vector & mu()const{return mu_;}
      const SpdMatrix & Sigma()const{return Sigma_;}
      std::ostream & print(std::ostream &out)const;

     private:
      Vector mu_;
      SpdMatrix Sigma_;
    };

    //----------------------------------------------------------------------
    class InverseWishartPrior {
     public:
      explicit InverseWishartPrior(double variance_guess_weight, const SpdMatrix &variance_guess);
      double variance_guess_weight() const {return variance_guess_weight_;}
      const SpdMatrix & variance_guess() const {return variance_guess_;}
     private:
      double variance_guess_weight_;
      SpdMatrix variance_guess_;
    };
    //----------------------------------------------------------------------
    class NormalInverseWishartPrior {
     public:
      explicit NormalInverseWishartPrior(const Vector &mu_guess, double mu_guess_weight, const SpdMatrix &sigma_guess, double sigma_guess_weight);
      const Vector & mu_guess()const{return mu_guess_;}
      double mu_guess_weight()const{return mu_guess_weight_;}
      const SpdMatrix & Sigma_guess()const{return sigma_guess_;}
      double Sigma_guess_weight()const{return sigma_guess_weight_;}
      std::ostream & print(std::ostream &out)const;

     private:
      Vector mu_guess_;
      double mu_guess_weight_;
      SpdMatrix sigma_guess_;
      double sigma_guess_weight_;
    };

    //----------------------------------------------------------------------
    class MvnIndependentSigmaPrior {
     public:
      explicit MvnIndependentSigmaPrior(const MvnPrior &mu_prior, std::vector<SdPrior> sigma_priors);
      const MvnPrior & mu_prior()const{return mu_prior_;}
      const SdPrior & sigma_prior(int i)const{return sigma_priors_[i];}

     private:
      MvnPrior mu_prior_;
      std::vector<SdPrior> sigma_priors_;
    };

    //----------------------------------------------------------------------
    class MvnDiagonalPrior {
     public:
      explicit MvnDiagonalPrior(const Vector &mean, const Vector &sd);
      const Vector & mean()const{return mean_;}
      const Vector & sd()const{return sd_;}

     private:
      Vector mean_;
      Vector sd_;
    };

    //----------------------------------------------------------------------
    // A discrete prior over the integers {lo, ..., hi}.
    class DiscreteUniformPrior {
     public:
      explicit DiscreteUniformPrior(int lo, int hi);
      double logp(int value) const;
      int lo() const {return lo_;}
      int hi() const {return hi_;}

     private:
      int lo_, hi_;
      double log_normalizing_constant_;
    };


    // A poisson prior, potentially truncated to the set {lo, ..., hi}.
    class PoissonPrior {
     public:
      explicit PoissonPrior(double lambda, double lo, double hi);
      double logp(int value) const;
      double lambda() const {return lambda_;}

     private:
      double lambda_;
      double lo_, hi_;
      double log_normalizing_constant_;
    };

    class PointMassPrior {
     public:
      explicit PointMassPrior(int location);
      double logp(int value) const;
      int location() const {return location_;}

     private:
      int location_;
    };

    //----------------------------------------------------------------------
    // This class is for handling spike and slab priors where there is
    // no residual variance parameter.  See the R help files for
    // SpikeSlabPrior or IndependentSpikeSlabPrior.
    class SpikeSlabGlmPrior {
     public:
      // Args:
      //   r_prior: An R object inheriting from SpikeSlabPriorBase.
      //     Elements of 'prior' relating to the residual variance are
      //     ignored.  If 'prior' inherits from
      //     IndependentSpikeSlabPrior then the slab will be an
      //     IndependentMvnModel.  Otherwise it will be an MvnModel.
      explicit SpikeSlabGlmPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips);
      virtual ~SpikeSlabGlmPrior() {}
      const Vector &prior_inclusion_probabilities() {
        return prior_inclusion_probabilities_;
      }
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<MvnBase> slab() {return slab_;}
      int max_flips() const {return max_flips_;}

     private:
      Vector prior_inclusion_probabilities_;
      Ptr<VariableSelectionPrior> spike_;
      Ptr<MvnBase> slab_;
      int max_flips_;
    };

    //----------------------------------------------------------------------
    // This is for the standard Zellner G prior in the regression
    // setting.  See the R help files for SpikeSlabPrior.
    class RegressionConjugateSpikeSlabPrior {
     public:
      // Args:
      //   r_prior: The R object containing the information needed to
      //     construct the prior.
      //   residual_variance: The residual variance parameter from the
      //     regression model described by the prior.
      RegressionConjugateSpikeSlabPrior(
        const Vector &prior_inclusion_probabilities,
        const Vector &prior_mean, const SpdMatrix &prior_precision, 
        const Vector &prior_variance_diagonal, int max_flips,
        double prior_df, double sigma_guess, double sigma_upper_limit,
        const Ptr<UnivParams> &residual_variance);
      const Vector &prior_inclusion_probabilities() {
        return prior_inclusion_probabilities_;}
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<MvnGivenScalarSigmaBase> slab() {return slab_;}
      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      int max_flips() const {return max_flips_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Vector prior_inclusion_probabilities_;
      Ptr<VariableSelectionPrior> spike_;
      Ptr<MvnGivenScalarSigmaBase> slab_;
      Ptr<ChisqModel> siginv_prior_;
      int max_flips_;
      double sigma_upper_limit_;
    };

    //----------------------------------------------------------------------
    // A version of the RegressionConjugateSpikeSlabPrior for
    // regression models with Student T errors.
     class StudentRegressionConjugateSpikeSlabPrior
         : public RegressionConjugateSpikeSlabPrior {
      public:
       StudentRegressionConjugateSpikeSlabPrior(
        const Vector &prior_inclusion_probabilities,
        const Vector &prior_mean, const SpdMatrix &prior_precision, 
        const Vector &prior_variance_diagonal, int max_flips,
        double prior_df, double sigma_guess, double sigma_upper_limit,
        Ptr<DoubleModel> df_prior_model, const Ptr<UnivParams> &residual_variance);
       Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

      private:
       Ptr<DoubleModel> df_prior_;
     };


    //----------------------------------------------------------------------
    // This is for the standard Zellner G prior in the regression
    // setting.  See the R help files for SpikeSlabPrior or
    // IndependentSpikeSlabPrior.
    class RegressionNonconjugateSpikeSlabPrior
        : public SpikeSlabGlmPrior {
     public:
      // Use this constructor if the prior variance is independent of
      // the residual variance.
      // Args:
      //   prior:  An R list containing the following objects
      //   - prior.inclusion.probabilities: Vector of prior inclusion
      //       probabilities.
      //   - mu:  Prior mean given inclusion.
      //   - siginv: Either a vector of prior precisions (for the
      //       independent case) or a positive definite matrix giving
      //       the posterior precision of the regression coefficients
      //       given inclusion.
      //   - prior.df: The number of observations worth of weight to
      //       be given to sigma.guess.
      //   - sigma.guess:  A guess at the residual variance
      explicit RegressionNonconjugateSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips, 
          double prior_df, double sigma_guess, double sigma_upper_limit);

      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Ptr<ChisqModel> siginv_prior_;
      double sigma_upper_limit_;
    };
    //----------------------------------------------------------------------
    class ArSpikeSlabPrior
        : public RegressionNonconjugateSpikeSlabPrior {
     public:
      explicit ArSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips, 
          double prior_df, double sigma_guess, double sigma_upper_limit, 
          bool truncate);
      bool truncate() const {return truncate_;}

     private:
      bool truncate_;
    };

    //----------------------------------------------------------------------
    // A version of RegressionNonconjugateSpikeSlabPrior for
    // regression models with Student T errors.
    class StudentRegressionNonconjugateSpikeSlabPrior
        : public RegressionNonconjugateSpikeSlabPrior {
     public:
      explicit StudentRegressionNonconjugateSpikeSlabPrior(const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const SpdMatrix &prior_precision, int max_flips, 
          double prior_df, double sigma_guess, double sigma_upper_limit,
          Ptr<DoubleModel> df_prior_model);
      Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

     private:
      Ptr<DoubleModel> df_prior_;
    };

    //----------------------------------------------------------------------
    // Use this class for the Clyde and Ghosh data augmentation scheme
    // for regression models.  See the R help files for
    // IndependentSpikeSlabPrior.
    class IndependentRegressionSpikeSlabPrior {
     public:
      IndependentRegressionSpikeSlabPrior(
          const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const Vector &prior_variance_diagonal,
          int max_flips, double prior_df, double sigma_guess, double sigma_upper_limit,
          const Ptr<UnivParams> &sigsq);
      const Vector &prior_inclusion_probabilities() {
        return prior_inclusion_probabilities_;}
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<IndependentMvnModelGivenScalarSigma> slab() {return slab_;}
      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      int max_flips() const {return max_flips_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Vector prior_inclusion_probabilities_;
      Ptr<VariableSelectionPrior> spike_;
      Ptr<IndependentMvnModelGivenScalarSigma> slab_;
      Ptr<ChisqModel> siginv_prior_;
      int max_flips_;
      double sigma_upper_limit_;
    };

    //----------------------------------------------------------------------
    // A version of IndependentRegressionSpikeSlabPrior for regression
    // models with Student T errors.
    class StudentIndependentSpikeSlabPrior
        : public IndependentRegressionSpikeSlabPrior {
     public:
      StudentIndependentSpikeSlabPrior(
          const Vector &prior_inclusion_probabilities,
          const Vector &prior_mean, const Vector &prior_variance_diagonal,
          int max_flips, double prior_df, double sigma_guess, double sigma_upper_limit,
          Ptr<DoubleModel> df_prior_model, const Ptr<UnivParams> &sigsq);
      Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

     private:
      Ptr<DoubleModel> df_prior_;
    };

    //----------------------------------------------------------------------
    // Conjugate prior for regression coefficients.  Multivariate
    // normal given sigma^2 and X.
    class RegressionCoefficientConjugatePrior {
     public:
      explicit RegressionCoefficientConjugatePrior(const Vector &mean, double sample_size, const Vector &additional_prior_precision, double diagonal_weight);
      const Vector &mean() const {return mean_;}
      double sample_size() const {return sample_size_;}
      const Vector &additional_prior_precision() const {
        return additional_prior_precision_;
      }
      double diagonal_weight() const {return diagonal_weight_;}

     private:
      Vector mean_;
      double sample_size_;
      Vector additional_prior_precision_;
      double diagonal_weight_;
    };

    class UniformPrior {
     public:
      explicit UniformPrior(double lo, double hi, double initial_value);
      double lo() const {return lo_;}
      double hi() const {return hi_;}
      double initial_value() const {return initial_value_;}
     private:
      double lo_, hi_;
      double initial_value_;
    };

    inline std::ostream & operator<<(std::ostream &out, const NormalPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const SdPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const BetaPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const MarkovPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out,
                                     const ConditionalNormalPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const MvnPrior &p) {
      return p.print(out); }

    Ptr<LocationScaleDoubleModel> create_double_model(std::shared_ptr<DoublePrior> spec);


  }  // namespace pybsts
}  // namespace BOOM

#endif // BOOM_PYBSTS_PRIOR_SPECIFICATION_HPP_
