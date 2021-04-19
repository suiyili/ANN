#ifdef TEST

#include "multi_channel_layer_factory.hpp"
#include "layer_mock.hpp"
#include "neuron_factory_mock.hpp"
#include <catch2/catch_test_macros.hpp>

namespace cnn::net {

SCENARIO("multi-channel layer factory test", "[multi_channel_layer_factory]") {
  GIVEN("a previous layer with shape size less than the unit shape") {
    layer_mock prev_layer({3});
    tensor_shape unit_shape{3, 3};
    WHEN("create multi channel layer") {
      neuron_factory_mock factory;
      multi_channel_layer_factory creator(unit_shape, 2, factory);
      THEN("it will throw") {
        auto ctor = [&prev_layer, &creator] { creator.create(prev_layer); };
         REQUIRE_THROWS_AS(ctor(), std::runtime_error);
      }
    }
  }
}

} // namespace cnn::net

#endif // TEST