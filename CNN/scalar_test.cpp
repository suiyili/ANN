#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "scalar.hpp"
#include "test_value.hpp"

using namespace Catch;

namespace cnn::layer {

SCENARIO("scalar test", "[scalar]") {
  GIVEN("a scalar neuron") {
    scalar s;
    
    AND_GIVEN("a proper ratio") {
      constexpr float ratio = .25f;
      WHEN("it drops out") {
        s.dropout(ratio);
        THEN("it should convert it to scalar value") {
          constexpr auto expected = 1.0f / (1.0f - ratio);
          REQUIRE((float)s == Approx(expected).margin(test_precision));
        }
      }
      AND_GIVEN("a size") {
        constexpr size_t size = 100U;
        WHEN("dropout") {
          s.dropout(ratio);
          AND_WHEN("get scaled size") {
            auto actual = s.scaled(size);
            THEN("it should scale the given size") {
              constexpr float expected = size * (1.0f - ratio);
              REQUIRE(actual == (size_t)expected);
            }
          }
        }
      }
    }
    AND_GIVEN("a zero ratio") {
      constexpr float ratio = .0f;
      AND_GIVEN("a size") {
        constexpr size_t size = 200U;
        WHEN("dropout") {
          s.dropout(ratio);
          AND_WHEN("get scaled size") {
            auto actual = s.scaled(size);
            THEN("it should get full size") { REQUIRE(actual == size); }
          }
        }
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST