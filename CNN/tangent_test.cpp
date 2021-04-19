#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "tangent.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("tangent test", "[tangent]") {
  GIVEN("a tangent neuron") {
    auto inputs = std::make_unique<layer_mock>(1U);
    value_factory_mock weights({1.0f, 1.0f});
    tangent tn(std::move(inputs), weights);
    WHEN("get result by given z") {
      const float z = 3.14f;
      auto result = tn.get_result(z);
      THEN("its output is tanh") {
        auto output = tanh(z);
        REQUIRE(result.output == Approx(output).margin(test_precision));
        AND_THEN("its derivative is base on tahn derivation") {
          auto derivative = (1.0f - output) * (1.0f + output);
          REQUIRE(result.derivative ==
                  Approx(derivative).margin(test_precision));
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST