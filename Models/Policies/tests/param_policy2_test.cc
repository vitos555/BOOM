#include "gtest/gtest.h"
#include "Models/GaussianModel.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ParamPolicy2Test : public ::testing::Test {
   protected:
    ParamPolicy2Test() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ParamPolicy2Test, ParamVectorTest) {
    GaussianModel model1(1.2, 3.7);
    EXPECT_EQ(2, model1.parameter_vector().size());
    model1.disable_parameter_1();

    model1.set_mu(0.1);
    EXPECT_DOUBLE_EQ(0.1, model1.mu());
    EXPECT_DOUBLE_EQ(2.7, model1.sigma());

    EXPECT_EQ(1, model1.parameter_vector().size());
    
  }
  
}  // namespace
