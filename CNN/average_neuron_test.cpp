#ifdef TEST

#include "average_neuron.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("average neuron test", "[average_neuron]") {
  GIVEN("an average neuron") {
    auto inputs = std::make_unique<layer_mock>(3U);

    const value_array values{0.25f, .33f, .50f};
    auto &inputs_v = *inputs;
    for (size_t i = 0; i < values.size(); ++i)
      inputs_v[i].set_output(values[i]);
    average_neuron average(std::move(inputs));

    WHEN("forward") {
      average.forward();
      THEN("it should output average of inputs") {
        auto expected = values.sum() / values.size();
        REQUIRE(average.output() == Approx(expected).margin(test_precision));
      }
      AND_WHEN("transfer propagate back") {
        const float error = .25f;
        average.propagate_back(error);
        THEN("it should propagate back to source") {
          auto expected = error / values.size();

          for (size_t i = 0; i < inputs_v.size(); ++i) {
            auto source_called = inputs_v[i].propagate_back_called_with();
            REQUIRE(source_called == Approx(expected).margin(test_precision));
          }
        }
      }
      AND_WHEN("train") {
        average.train();
        THEN("it should not transfer train") {
          for (size_t i = 0; i < inputs_v.size(); ++i)
            REQUIRE_FALSE(inputs_v[i].train_called());
        }
      }
      AND_WHEN("learn") {
        const float learning_rate = .2f;
        average.learn(learning_rate);
        THEN("it should transfer the learn") {

          for (size_t i = 0; i < inputs_v.size(); ++i) {
            auto source_called = inputs_v[i].learn_called_with();
            REQUIRE_FALSE(source_called == learning_rate);
          }
        }
      }
    }
  }
}

} // namespace cnn::neuron
#endif // TEST