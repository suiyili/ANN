#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "hidden_neuron_mock.hpp"
#include "layer_mock.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("hidden neuron test", "[hidden_neuron]") {
  GIVEN("a hidden neuron based on input layer") {

    auto inputs = std::make_unique<layer_mock>(1U);
    auto &inputs_v = *inputs;

    value_factory_mock factory({1.0f, 1.0f});
    hidden_neuron_mock n(std::move(inputs), factory);

    WHEN("set argument") {
      const float z = 5.28f;
      n.set_argument(z);
      THEN("it should call get result with the arg") {
        REQUIRE(n.call_get_result_with() == Approx(z).margin(test_precision));
      }
    }
    WHEN("set output") {
      const float out = 5.28f;
      n.set_output(out);
      n.set_argument(0.0f);
      THEN("it should get the setting as output") {
        REQUIRE(n.output() == Approx(out).margin(test_precision));
      }
    }

    WHEN("train with back propagation") {
      
      const float deri = 5.28f;
      const float error_gradient = 2.3f;

      n.set_derivative(deri);
      n.set_argument(0.0f);
      n.propagate_back(error_gradient);
      n.train();

      THEN("it should update input layer with propagation data") {
        REQUIRE(inputs_v[0].propagate_back_called_with() ==
                Approx(error_gradient * deri).margin(test_precision));
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST