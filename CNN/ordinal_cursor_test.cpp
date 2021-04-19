#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "ordinal_cursor.hpp"
#include "tensor_vector_matcher.hpp"

namespace cnn {

SCENARIO("ordinal cursor test", "[ordinal_cursor]") {

  GIVEN("an zero index as init") {
    auto init_with_zero = [] { ordinal_cursor counter{0, 0}; };
    WHEN("construct ordinal cursor") {
      THEN("it will throw") {
        REQUIRE_THROWS_AS(init_with_zero(), std::invalid_argument);
      }
    }
  }

  GIVEN("empty as init") {
    auto init_empty = [] { ordinal_cursor counter1({}); };
    WHEN("construct ordinal cursor") {
      THEN("it should throw") {
        REQUIRE_THROWS_AS(init_empty(), std::invalid_argument);
      }
    }
  }

  GIVEN("a ordinal cursor") {
    ordinal_cursor counter{2, 2};
    WHEN("increase") {
      auto actual = counter.increase();
      THEN("it will return prev saved value") {
        tensor_vector expected{0, 0};
        REQUIRE_THAT(actual, Equals(expected));
      }
    }
    AND_WHEN("increasing not meet the lowest level limit") {
      counter.increase();
      auto actual = counter.increase();
      THEN("it only increase lowest level") {
        tensor_vector expected{1, 0};
        REQUIRE_THAT(actual, Equals(expected));
      }
    }
    AND_WHEN("increase meets lowest level limit") {
      counter.increase();
      counter.increase();
      auto actual = counter.increase();
      THEN("it will push one to higher level") {
        tensor_vector expected{0, 1};
        REQUIRE_THAT(actual, Equals(expected));
      }
    }

    AND_WHEN("increase meets highest level limit") {
      counter.increase();
      counter.increase();
      counter.increase();
      counter.increase();
      THEN("increase will get back to zero") {
        tensor_vector expected{0, 0};
        REQUIRE_THAT(counter.increase(), Equals(expected));
      }
    }
  }
}

} // namespace cnn

#endif // TEST