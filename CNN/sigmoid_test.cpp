#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "sigmoid.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("sigmoid test", "[sigmoid]") {
  GIVEN("a sigmoid") {
    auto inputs = std::make_unique<layer_mock>(1U);
    value_factory_mock repo({1.0f, 1.0f});
    sigmoid s(std::move(inputs), repo);
    WHEN("get result with given z") {
      const float z = 3.45f;
      auto result = s.get_result(z);
      THEN("its output is based on sigmoid func") {
        auto output = 1.0f / (1.0f + exp(-z));
        REQUIRE(result.output == Approx(output).margin(test_precision));
        AND_THEN("its derivative is base on sigmoid func") {
          auto derivative = output * (1.0f - output);
          REQUIRE(result.derivative ==
                  Approx(derivative).margin(test_precision));
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST