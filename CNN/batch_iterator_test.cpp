#ifdef TEST

#include "batch_iterator.hpp"
#include <catch2/catch_test_macros.hpp>

namespace cnn {

SCENARIO("batch iterator test", "[batch_iterator]") {

  GIVEN("a series indices") {
    const data_indices source{2, 4, 6, 8, 1, 3, 5, 7};

    AND_GIVEN("a undividable batch size") {
      size_t batch_size = 3U;
      WHEN("init batch") {
        auto init = [&source, &batch_size] {
          batch_iterator batch(source, batch_size);
        };
        THEN("it should throw") {
          REQUIRE_THROWS_AS(init(), std::invalid_argument);
        }
      }
    }
    AND_GIVEN("a proper batch size") {
      size_t batch_size = 4U;
      WHEN("it moves to next batch") {
        batch_iterator batch(source, batch_size);
        CHECK(batch.move_next());
        THEN("its index will start from the next batch") {
          for (size_t i = 0U; i < batch.size(); ++i)
            REQUIRE(batch[i] == source[batch.size() + i]);
        }
        AND_THEN("it can't move anymore") { REQUIRE_FALSE(batch.move_next()); }
      }
    }
  }
}

} // namespace cnn

#endif // TEST