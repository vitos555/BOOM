// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef BOOM_CUMULATIVE_PROBIT_SAMPLER_HPP_
#define BOOM_CUMULATIVE_PROBIT_SAMPLER_HPP_

#include <Models/Glm/CumulativeProbitModel.hpp>
#include <Models/Glm/RegressionModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM{

  class CumulativeProbitSampler
      : public PosteriorSampler
  {
   public:
    CumulativeProbitSampler(CumulativeProbitModel *m,
                            const Ptr<MvnBase> &beta_prior,
                            RNG &seeding_rng = GlobalRng::rng);

    void impute_latent_data();
    void draw_beta();
    void draw_delta();
    void draw() override;
    double logpri() const override;
   private:
    CumulativeProbitModel *m_;
    Ptr<MvnBase> beta_prior_;
    NeRegSuf suf_;
    SpdMatrix ivar_;
    Vector mu_;
    Vector beta_;
    Vector delta_;
    // assume a flat prior on delta
  };
}
#endif// BOOM_CUMULATIVE_PROBIT_SAMPLER_HPP_
