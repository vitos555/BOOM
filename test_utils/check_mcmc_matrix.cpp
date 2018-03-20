/*
  Copyright (C) 2018 Steven L. Scott

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

#include "test_utils/test_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  std::string CheckMatrixStatus::error_message() const {
    std::ostringstream err;
    err << "Too many columns of 'draws' failed to cover true values." << endl
        << "Failure rate: " << fraction_failing_to_cover * 100 << " (%) " << endl
        << "Rate limit: " << failure_rate_limit * 100 << " (%) " << endl;
    return err.str();
  }
  
  CheckMatrixStatus check_mcmc_matrix(
      const Matrix &draws,
      const Vector &truth,
      double confidence,
      bool control_multiple_comparisons) {
    if (confidence <= 0 || confidence >= 1) {
      report_error("Confidence must be strictly between 0 and 1.");
    }
    if (confidence < .5) confidence = 1 - confidence;
    double alpha = 1 - confidence;
    double alpha_2 = .5 * alpha;

    CheckMatrixStatus status;
    for (int i = 0; i < ncol(draws); ++i) {
      Vector v = sort(draws.col(i));
      double lower = sorted_vector_quantile(v, alpha_2);
      double upper = sorted_vector_quantile(v, 1 - alpha_2);
      bool covers = lower <= truth[i] && upper >= truth[i];
      if (!covers) {
        ++status.fails_to_cover;
      }
    }

    double fraction_failing_to_cover = status.fails_to_cover;
    fraction_failing_to_cover /= ncol(draws);
    double coverage_rate_limit = confidence;
    if (control_multiple_comparisons) {
      double se = sqrt(confidence * (1 - confidence) / ncol(draws));
      coverage_rate_limit -= 2 * se;
    }
    status.failure_rate_limit = 1 - coverage_rate_limit;
    if (fraction_failing_to_cover >= status.failure_rate_limit) {
      status.ok = false;
      status.fraction_failing_to_cover = fraction_failing_to_cover;
    }
    return status;
  }

  bool CheckMcmcVector(const Vector &draws, double truth, double confidence) {
    if (confidence <= 0 || confidence >= 1) {
      report_error("Confidence must be strictly between 0 and 1.");
    }
    if (confidence < .5) confidence = 1 - confidence;
    double alpha = 1 - confidence;
    double alpha_2 = .5 * alpha;
    Vector v = sort(draws);
    double lo = sorted_vector_quantile(v, alpha_2);
    double hi = sorted_vector_quantile(v, 1 - alpha_2);
    return lo <= truth && hi >= truth;
  }
  
}  // namespace BOOM