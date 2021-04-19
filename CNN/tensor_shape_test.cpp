#ifdef TEST

#include "tensor_shape.hpp"
#include "tensor_vector_matcher.hpp"
#include <catch2/catch_test_macros.hpp>

namespace cnn {

SCENARIO("tensor shape test", "[tensor_shape]") {

  GIVEN("a tensor shape") {
    tensor_shape shape{2, 3, 4};
    WHEN("get its volume") {
      auto volume = shape.volume();
      THEN("it should get all fields multiplication") {
        REQUIRE(volume == 2U * 3U * 4U);
      }
    }
    WHEN("get its dimension") { 
      auto dim = shape.dimensions();
      THEN("it should get number of fields") {
        REQUIRE(dim == 3U);
      }
    }
  }

  GIVEN("a tensor shape") { 
    tensor_shape shape{20, 30};
    AND_GIVEN("a pixel in the shape") { 
      tensor_vector p{19, 14};
      WHEN("check if it includes the pixel") {
        bool included = shape.includes(p);
        THEN("it should get true") { 
          REQUIRE(included);
        }
      }
    }
    AND_GIVEN("a pixel not in the shape") {
      tensor_vector p{23, 14};
      WHEN("check if it includes the pixel") {
        bool included = shape.includes(p);
        THEN("it should get true") { 
          REQUIRE_FALSE(included); }
      }
    }
  }

  GIVEN("a tensor shape") {
    tensor_shape shape{20, 30, 3};
    AND_GIVEN("an invalid pixel") {
      tensor_vector pixel{25, 14};
      WHEN("get ordinal of the pixel") {
        THEN("it should throw") {
          REQUIRE_THROWS_AS(shape.ordinal_of(pixel), std::invalid_argument);
        }
      }
    }
    AND_GIVEN("a out of range pixel") {
      tensor_vector pixel{25, 14, 1};
      WHEN("get ordinal of the pixel") {
        THEN("it should throw") {
          REQUIRE_THROWS_AS(shape.ordinal_of(pixel), std::range_error);
        }
      }
    }
    AND_GIVEN("a valid pixel in the range of tensor shape") {
      tensor_vector pixel{15, 14, 1};
      WHEN("get ordinal of the pixel") {

        auto ordinal = shape.ordinal_of(pixel);
        THEN("it should return the ordinal") {
          size_t expected = 30U * 20U * 1U + 14U * 20U + 15U;
          REQUIRE(ordinal == expected);
        }
      }
    }
  }

  GIVEN("a tensor shape with 1 as field") { 
      tensor_shape shape{20, 1, 30, 1};
      WHEN("shrink it") { 
          auto reduced = shape.shrink();
          THEN("it should remove 1 field") {
            tensor_shape expected{20, 30};
            REQUIRE_THAT(reduced, Equals(expected));
        }
    }
  }

  GIVEN("a tensor shape") {
    tensor_shape shape{20U, 30U};
    WHEN("inflate it without param") {
      auto inflated_back = shape.inflate();
      THEN("it should inflate one") {
        tensor_shape expected_back{20U, 30U, 1U};
        REQUIRE_THAT(inflated_back, Equals(expected_back));
      }
    }
    WHEN("inflate with 0") {
      auto inflated_front = shape.inflate(0U);
      THEN("it should inflate at 0 position") {
        tensor_shape expected_front{1U, 20U, 30U};
        REQUIRE_THAT(inflated_front, Equals(expected_front));
      }
    }

    WHEN("inflate with 1") {
      auto inflated_middle = shape.inflate(1U);
      THEN("it should inflate at 1 position") {
        tensor_shape expected_middle{20U, 1U, 30U};
        REQUIRE_THAT(inflated_middle, Equals(expected_middle));
      }
    }
  }

}
} // namespace cnn
#endif // TEST