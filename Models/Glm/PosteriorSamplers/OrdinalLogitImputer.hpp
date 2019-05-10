#ifndef BOOM_GLM_ORDINAL_LOGIT_IMPUTER_HPP_
#define BOOM_GLM_ORDINAL_LOGIT_IMPUTER_HPP_
/*
  Copyright (C) 2005-2019 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "distributions/rng.hpp"

namespace BOOM {

  class OrdinalLogitImputer {
   public:
    // Impute the latent variable Z, where Z ~ logistic(eta), conditional on Z
    // between lower_cutpoint and upper_cutpoint.
    static double impute(RNG &rng, double eta, double lower_cutpoint,
                         double upper_cutpoint);
  };
  
}  // namespace BOOM

#endif //  BOOM_GLM_ORDINAL_LOGIT_IMPUTER_HPP_
