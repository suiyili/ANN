#ifdef TEST

#include "input_neuron.hpp"
#include <catch2/catch_test_macros.hpp>

namespace cnn::neuron {

SCENARIO("input neuron test", "[input_neuron]") {
  GIVEN("input source") {
    const float expected1 = .38f;
    const float expected2 = .66f;
    const value_array source1{.25f, expected1, .11f, .90f};
    const value_array source2{.25f, expected2, .11f, .90f};
    const value_array *source = &source1;
    WHEN("construct input nueron") {
      input_neuron n(source, 1U);
      THEN("its output is same as source") { REQUIRE(n.output() == expected1); }

      AND_WHEN("change the source pointer") {
        source = &source2;
        THEN("input neuron will point to new source") {
          REQUIRE(n.output() == expected2);
        }
      }
    }
  }
}

} // namespace cnn::neuron
#endif // TEST