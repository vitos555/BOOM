// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#include <cmath>
#include "distributions.hpp"
namespace BOOM {
  int random_int(int lo, int hi) {
    return random_int_mt(GlobalRng::rng, lo, hi);
  }

  int random_int_mt(RNG& rng, int lo, int hi) {
    double tmp = runif_mt(rng, lo, hi + 1);
    return static_cast<int>(std::floor(tmp));
  }
}  // namespace BOOM
