#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "inflation_layer.hpp"
#include "layer_mock.hpp"
#include "tensor_vector_matcher.hpp"

namespace cnn::layer {

SCENARIO("inflation layer test", "[inflation_layer]") {

  GIVEN("a prev layer shape not match expand shape") {
    tensor_shape prev_shape{2, 3, 4};

    layer_mock prev_layer(prev_shape);
    AND_GIVEN("a expand shape which does not match") {
      tensor_shape expand{2, 2, 2};
      WHEN("construct inflation layer") {
        auto ctor = [&prev_layer, &expand] {
          inflation_layer layer(prev_layer, expand);
        };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(ctor(), std::invalid_argument);
        }
      }
    }
  }

  GIVEN("a certian shape prev layer") {
    tensor_shape prev_shape{2, 2};
    layer_mock prev_layer(prev_shape);
    AND_GIVEN("a inflation shape to build inflation layer") {
      tensor_shape to_inflate{2, 2, 5};

      WHEN("build infaltion layer") {
        inflation_layer layer(prev_layer, to_inflate);
        THEN("its shape should be expanded") {
          tensor_shape expected{2 * 2, 2 * 2, 5};
          REQUIRE_THAT(layer.layer_shape(), Equals(expected));
        }
        THEN("its indices corresponds to prev layer indices") {
          REQUIRE(&layer[0] == &prev_layer[0]);
          REQUIRE(&layer[5] == &prev_layer[0]);
          REQUIRE(&layer[17] == &prev_layer[0]);

          REQUIRE(&layer[2] == &prev_layer[1]);
          REQUIRE(&layer[7] == &prev_layer[1]);
          REQUIRE(&layer[22] == &prev_layer[1]);

          REQUIRE(&layer[8] == &prev_layer[2]);
          REQUIRE(&layer[9] == &prev_layer[2]);
          REQUIRE(&layer[28] == &prev_layer[2]);

          REQUIRE(&layer[10] == &prev_layer[3]);
          REQUIRE(&layer[14] == &prev_layer[3]);
          REQUIRE(&layer[31] == &prev_layer[3]);
        }
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST