#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "max_neuron.hpp"
#include "test_value.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("max neuron test", "[max_neuron]") {
  GIVEN("a max neuron") {
    const size_t max_id = 1U;
    auto inputs = std::make_unique<layer_mock>(3);

    auto &inputs_v = *inputs;
    inputs_v[max_id].set_output(.50f);

    max_neuron max_n(std::move(inputs));

    WHEN("forward") {
      max_n.forward();
      THEN("its output is same as the max one from source") {
        auto max = inputs_v[max_id].output();
        REQUIRE(max_n.output() == Approx(max).margin(test_precision));
      }
      AND_WHEN("propagate back") {
        constexpr float error = .25f;
        max_n.propagate_back(error);
        THEN("it will update source max neuron") {
          auto source_propagated =
              inputs_v[max_id].propagate_back_called_with();
          REQUIRE(source_propagated == Approx(error).margin(test_precision));
        }
      }
      AND_WHEN("train") {
        max_n.train();
        THEN("it will not transfer to source train") {
          REQUIRE_FALSE(inputs_v[max_id].train_called());
        }
      }
      AND_WHEN("learn") {
        constexpr float learning_rate = .3f;
        max_n.learn(learning_rate);
        THEN("it should transfer to source") {
          auto source_learned = inputs_v[max_id].learn_called_with();
          REQUIRE_FALSE(source_learned == learning_rate);
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST