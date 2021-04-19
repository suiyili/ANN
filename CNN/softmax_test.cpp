#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "softmax.hpp"
#include "neuron_mock.hpp"
#include "test_value.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("cross entropy neuron test", "[softmax]") {
  GIVEN("a input neuron") {
    const float last_output = .25f;
    neuron_mock input_neuron(last_output);
    AND_GIVEN("a cross entropy neuron on top of it") {
      softmax n(input_neuron);
      WHEN("set output") {
        const float output = .25f;
        n.set_output(output);
        THEN("its output is same as the given value") {
          REQUIRE(n.output() == output);
        }
        AND_WHEN("propagate error") {
          const float error = .85f;
          n.propagate_back(error);
          THEN("it should pass error corrected output back") {
            auto expected = output - error;
            REQUIRE(input_neuron.propagate_back_called_with() ==
                    Approx(expected).margin(test_precision));
          }
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST