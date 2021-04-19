#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "hidden_layer.hpp"
#include "neuron_factory_mock.hpp"
#include "tensor_vector_matcher.hpp"

namespace cnn::layer {

SCENARIO("hidden layer test", "[hidden_layer]") {

  GIVEN("a hidden layer") {
    const tensor_shape shape{2, 2};
    neuron_factory_mock factory;
    hidden_layer hidden(shape, factory);
    net_layer_i &layer = hidden;

    WHEN("forward") {
      layer.forward();
      THEN("it should call each neuron with forward value") {
        for (size_t i = 0; i < shape.volume(); ++i) {
          auto &n = factory.get_neuron(i);
          REQUIRE(n.forward_called());
        }
      }
    }

    WHEN("train") {
      layer.train();
      THEN("it should train each neuron") {
        for (size_t i = 0; i < shape.volume(); ++i) {
          auto &n = factory.get_neuron(i);
          REQUIRE(n.train_called());
        }
      }
    }

    WHEN("learn") {
      const float learning_rate = .2f;
      layer.learn(learning_rate);
      THEN("it should call learn on each neuron") {}
      for (size_t i = 0; i < shape.volume(); ++i) {
        auto &n = factory.get_neuron(i);
        REQUIRE(n.learn_called_with() == learning_rate);
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST