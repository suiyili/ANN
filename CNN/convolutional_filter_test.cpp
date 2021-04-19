#ifdef TEST

#include <catch2/catch_all.hpp>
#include "convolutional_filter.hpp"
#include "feature_shape_error.hpp"
#include "tensor_vector_matcher.hpp"

namespace cnn::layer {

SCENARIO("convolutional filter test", "[convolutional_filter]") {

  GIVEN("a prev shape") {
    const tensor_shape prev_shape{8, 8, 3};
    AND_GIVEN("a mis-match dimension feature shape") {
      const tensor_shape feature_shape{5, 2};
      WHEN("construct convolutional filter") {
        auto ctor = [&] {
          convolutional_filter filter(feature_shape, prev_shape);
        };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(ctor(), feature_shape_error);
        }
      }
    }
    AND_GIVEN("a range mis-match feature shape") {
      const tensor_shape feature_shape{5, 3, 9};
      WHEN("construct convolutional filter") {
        auto ctor = [&] {
          convolutional_filter filter(feature_shape, prev_shape);
        };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(ctor(), feature_shape_error);
        }
      }
    }
    AND_GIVEN("a proper feature shape") {
      const tensor_shape feature_shape{5, 4, 3};
      convolutional_filter filter(feature_shape, prev_shape);
      WHEN("get layer shape from filter") {
        auto actual = filter.layer_shape();
        THEN("it should calculate new layer shape based on convolutional "
             "feature filter algo") {
          tensor_shape expected{(size_t)8 - 5 + 1, (size_t)8 - 4 + 1, 1};

          REQUIRE_THAT(actual, Equals(expected));
        }
      }
    }
  }

  GIVEN("8*8 prev shape") {
    const tensor_shape prev_shape{8, 8};
    /* The 8*8 matrix item will be given value by their ordinal
    0,1,2,3...7
    8...
    ...
    ......62,63
    */
    AND_GIVEN("4*4 feature shape and a start point") {
      const tensor_shape feature_shape{4, 4};
      convolutional_filter filter(feature_shape, prev_shape);
      const tensor_vector pos{1, 1};

      WHEN("access the indices of the filter") {
        auto feature_indices = filter[pos];
        THEN("it should get 4*4 sub matrix from the given start point") {

          /*The expected proxy should get [4,4] sub-matrix
          start from 1st row, 1st column of above [8,8] matrix
          9, 10, 11, 12
          17, 18, 19, 20
          25, 26, 27, 28
          33, 34, 35, 36
          */
          REQUIRE(feature_indices.size() == feature_shape.volume());

          REQUIRE(feature_indices[feature_shape.ordinal_of({0, 0})] == 9U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({1, 1})] == 18U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({2, 2})] == 27U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({3, 3})] == 36U);

          REQUIRE(feature_indices[feature_shape.ordinal_of({0, 3})] == 33U);
          REQUIRE(feature_indices[feature_shape.ordinal_of({3, 0})] == 12U);
        }
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST