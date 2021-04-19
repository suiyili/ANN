#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_factory_mock.hpp"
#include "layer_mock.hpp"
#include "network.hpp"
#include "test_value.hpp"

using namespace Catch;

namespace cnn::net {

SCENARIO("network test", "[network]") {
  GIVEN("a input layer") {
    auto layer = std::make_unique<layer_mock>(4U);
    auto &last_layer = *layer;
    last_layer[0].set_output(0.1f);
    last_layer[1].set_output(0.95f);
    last_layer[2].set_output(0.33f);
    last_layer[3].set_output(0.44f);

    AND_GIVEN("a network based on the input layer") {
      layer_factory_mock factory(std::move(layer));

      network net({2, 2});
      WHEN("create layer") {
        net.create_layer(factory);
        AND_WHEN("predict") {
          const size_t highest = net.predict({.2f, .3f, .2f, .1f});
          THEN("input layer forward get called") {
            REQUIRE(last_layer.forward_called());
            AND_THEN("highest one is 1") {
              auto expected = 1U;
              REQUIRE(highest == expected);
            }
          }
        }
        AND_WHEN("train the network") {
          const size_t label = 2U;
          net.train({.2f, .1f, .1f, .4f}, label);
          THEN("last layer train get called") {
            REQUIRE(last_layer.train_called());
            AND_THEN("it should propagate back to each neuron in last layer") {
              for (size_t i = 0; i < last_layer.size(); ++i) {
                REQUIRE(last_layer[i].propagate_back_called_with() ==
                        Approx(i == label ? 1.0f : .0f)
                            .margin(test_precision));
              }
            }
          }
        }
        AND_WHEN("it learns") {
          const float learning_rate = 0.4f;
          net.learn(learning_rate);
          THEN("last layer learn get called with given learning rate") {
            REQUIRE(last_layer.learn_called_with() == learning_rate);
          }
        }
      }
    }
  }
}

} // namespace cnn::net

#endif // TEST