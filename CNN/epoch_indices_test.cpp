#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "epoch_indices.hpp"

namespace cnn {

SCENARIO("epoch indices test", "[epoch_indices]") {
  GIVEN("an epoch") {
    const size_t epoch_size = 8U;
    
    epoch_indices epoch(epoch_size);
    WHEN("it generates batch") {   
      const size_t batch_size = 4U;   
      auto batch = epoch.generator(batch_size);
      THEN("it should produce a batch including given size of indices") {
        REQUIRE(batch_size ==  batch.size());
        for (size_t i = 0U; i < batch_size; ++i)
          REQUIRE(batch[i] == i);
      }
      
    }
  }
}

} // namespace cnn
#endif // TEST