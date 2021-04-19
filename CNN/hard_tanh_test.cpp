#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "hard_tanh.hpp"
#include "layer_mock.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("hardtanh test", "[hard_tanh]") {

  GIVEN("a hard tanh neuron") {
    auto inputs = std::make_unique<layer_mock>(1U);
    value_factory_mock factory({0.5f, 0.1f});
    hard_tanh ht(std::move(inputs), factory);

    AND_GIVEN("a z less than -1") {
      const float z = -3.45f;
      WHEN("get result from z") {
        auto result = ht.get_result(z);
        THEN("it should get output as -1") {
          auto output = -1.0f;
          REQUIRE(result.output == Approx(output).margin(test_precision));
        }
        THEN("it should get derivative 0") {
          auto derivative = 0.0f;
          REQUIRE(result.derivative ==
                  Approx(derivative).margin(test_precision));
        }
      }
    }

    AND_GIVEN("a z more than 1") {
      const float z = 3.45f;
      WHEN("get result from z") {
        auto result = ht.get_result(z);
        THEN("it should get output as 1") {
          auto output = 1.0f;
          REQUIRE(result.output == Approx(output).margin(test_precision));
        }
        THEN("it should get derivative 0") {
          auto derivative = 0.0f;
          REQUIRE(result.derivative ==
                  Approx(derivative).margin(test_precision));
        }
      }
    }

    AND_GIVEN("a z between -1 and 1") {
      const float z = .45f;
      WHEN("get result from z") {
        auto result = ht.get_result(z);
        THEN("it should get output as z") {
          REQUIRE(result.output == Approx(z).margin(test_precision));
        }
        THEN("it should get derivative 1") {
          auto derivative = 1.0f;
          REQUIRE(result.derivative ==
                  Approx(derivative).margin(test_precision));
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST