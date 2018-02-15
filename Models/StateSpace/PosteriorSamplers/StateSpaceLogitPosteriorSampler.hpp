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

#ifndef BOOM_STATE_SPACE_LOGIT_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_LOGIT_POSTERIOR_SAMPLER_HPP_

#include "Models/StateSpace/StateSpaceLogitModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp"

namespace BOOM {
  class StateSpaceLogitPosteriorSampler
      : public StateSpacePosteriorSampler {
   public:

    // Args:
    //   model: The model for which posterior samples are desired.
    //     All state components should have posterior samplers
    //     assigned to them before 'model' is passed to this
    //     constructor.  Likewise, the observation model should have
    //     observation_model_sampler assigned to it before being
    //     passed here.
    //   observation_model_sampler: The posterior sampler for the
    //     logistic regression observation model.  A separate handle
    //     to this sampler is necessary because the latent data
    //     imputation and parameter draw steps must be controlled
    //     separately.
    StateSpaceLogitPosteriorSampler(
        StateSpaceLogitModel *model,
        const Ptr<BinomialLogitSpikeSlabSampler> &observation_model_sampler,
        RNG &seeding_rng = GlobalRng::rng);

    // Impute the latent Gaussian observations and variances at each
    // data point, conditional on the state, observed data, and model
    // parameters.
    void impute_nonstate_latent_data() override;

    // Clear the complete_data_sufficient_statistics for the logistic
    // regression model.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete_data_sufficient_statistics for the
    // logistic regression model by adding the latent data from
    // observation t.  This update is conditional on the contribution
    // of the state space portion of the model, which is stored in the
    // "offset" component of observation t.
    void update_complete_data_sufficient_statistics(int t);

   private:
    StateSpaceLogitModel *model_;
    Ptr<BinomialLogitSpikeSlabSampler> observation_model_sampler_;
    BinomialLogitCltDataImputer data_imputer_;
  };
}

#endif // BOOM_STATE_SPACE_LOGIT_POSTERIOR_SAMPLER_HPP_
