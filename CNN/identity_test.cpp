#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "identity.hpp"
#include "layer_mock.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

namespace cnn::neuron {

SCENARIO("identity test", "[identity]") {
  GIVEN("an identity neuron") {
    auto inputs = std::make_unique<layer_mock>(1U);
    value_factory_mock repo({1.0f, 1.0f});
    identity tn(std::move(inputs), repo);
    WHEN("get result with given z") {
      const float z = 3.14f;
      auto result = tn.get_result(z);
      THEN("it should output the given z") {
        auto output = z;
        REQUIRE(result.output == output);
      }
      AND_THEN("it should have 1 as derivative") {
        auto derivative = 1.0f;
        REQUIRE(result.derivative == derivative);
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST