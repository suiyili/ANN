#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "feature_shape_error.hpp"
#include "pooling_filter.hpp"
#include "tensor_vector_matcher.hpp"

namespace cnn::layer {

SCENARIO("pooling filter test", "[pooling_filter]") {

  GIVEN("a prev shape") {
    const tensor_shape prev_shape{8, 8, 4};
    AND_GIVEN("a mis-match dimension feature shape") {
      const tensor_shape feature_shape{5, 2};
      WHEN("construct pooling layer") {
        auto ctor = [&] { pooling_filter filter(feature_shape, prev_shape); };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(ctor(), feature_shape_error);
        }
      }
    }
    AND_GIVEN("a range same feature shape") {
      const tensor_shape feature_shape{2, 3, 4};
      WHEN("construct pooling layer") {
        auto ctor = [&] { pooling_filter filter(feature_shape, prev_shape); };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(ctor(), feature_shape_error);
        }
      }
    }
    AND_GIVEN("a proper feature shape to build pooling layer") {
      const tensor_shape feature_shape{2, 4, 2};
      pooling_filter filter(feature_shape, prev_shape);
      WHEN("get layer shape") {
        auto actual = filter.layer_shape();
        THEN("it should calculate new layer base on pooling feature") {
          // number of dimensions should be same
          // calculate new layer shape based on pooling feature filter algo.
          tensor_shape expected{(size_t)8 / 2, (size_t)8 / 4, (size_t)4 / 2};

          REQUIRE_THAT(actual, Equals(expected));
        }
      }
    }
  }

  GIVEN("a 8*8 prev shape") {
    const tensor_shape prev_shape{8, 8};
    /* The 8*8 matrix item will be given value by their ordinal
        0,1,2,3...7
        8...
        ...
        ......62,63
        */

    AND_GIVEN("a 4*4 feature shape and start point") {
      const tensor_shape feature_shape{4, 4};
      pooling_filter filter(feature_shape, prev_shape);
      const tensor_vector pos{1, 0};
      
      WHEN("access feature indices") {        
        auto feature_indices = filter[pos];
        THEN("it should get 4*4 sub-matrix start from the start point") {

          /*The expected proxy should get [4,4] sub-matrix
          start from 1st row, 1st column of above [8,8] matrix
          4, 5, 6, 7
          12, 13, 14, 15
          20, 21, 22, 23
          28, 29, 30, 31
          */

          REQUIRE(feature_indices.size() == feature_shape.volume());

          REQUIRE(feature_indices[feature_shape.ordinal_of({0, 0})] == 4U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({1, 1})] == 13U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({2, 2})] == 22U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({3, 3})] == 31U);

          REQUIRE(feature_indices[feature_shape.ordinal_of({0, 3})] == 28U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({3, 0})] == 7U);
        }
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST