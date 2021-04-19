#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "layer_mock.hpp"
#include "softmax_layer.hpp"
#include "tensor_vector_matcher.hpp"
#include "test_value.hpp"
#include "value_array_matcher.hpp"

using namespace Catch;

namespace cnn::layer {

SCENARIO("softmax layer test", "[softmax_layer]") {
  GIVEN("a prev layer") {
    const tensor_shape shape{2, 2};
    layer_mock prev_layer(shape);

    prev_layer[0].set_output(17.5f);
    prev_layer[1].set_output(35.2f);
    prev_layer[2].set_output(44.8f);
    prev_layer[3].set_output(83.0f);
    AND_GIVEN("a softmax layer") {
      softmax_layer layer(prev_layer);
      WHEN("access a invalid index") {
        CHECK_THAT(layer.layer_shape(), Equals(shape));
        THEN("it should throw") {
//          REQUIRE_THROWS_AS(layer[shape.volume()], std::out_of_range);
        }
      }
      WHEN("forward") {
        
        layer.forward();

        THEN("it should forward each neuron with softmax value") {
          value_array expected(shape.volume());
          for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = exp(prev_layer[i].output());
          auto sum = expected.sum();
          for (auto &e : expected)
            e /= sum;

          for (size_t i = 0; i < shape.volume(); ++i)
            REQUIRE(layer[i].output() ==
                    Approx(expected[i]).margin(test_precision));
        }
      }
    }
  }
} // namespace cnn::layer

} // namespace cnn::layer

#endif // TEST