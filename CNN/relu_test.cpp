#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "relu.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("relu test", "[relu]") {
  GIVEN("a relu nueron") {
    auto inputs = std::make_unique<layer_mock>(1U);
    value_factory_mock repo({1.0f, 1.0f});
    relu r(std::move(inputs), repo);
    AND_GIVEN("a negative z value") {
      const float z = -3.45f;
      WHEN("get result") {
        auto result = r.get_result(z);
        THEN("its output is zero") {
          REQUIRE(result.output == Approx(0.0).margin(test_precision));
        }
        THEN("its derivative is 0") {
          REQUIRE(result.derivative ==
                  Approx(0.0).margin(test_precision));
        }
      }
    }
    AND_GIVEN("a positive z value") {
      const float z = 3.45f;
      WHEN("get result") {
        auto result = r.get_result(z);
        THEN("its output is z") {
          REQUIRE(result.output == Approx(z).margin(test_precision));
        }
        THEN("its derivative is 1") {
          REQUIRE(result.derivative ==
                  Approx(1.0).margin(test_precision));
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST