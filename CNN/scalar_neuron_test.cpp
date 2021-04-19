#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "neuron_mock.hpp"
#include "scalar_neuron.hpp"
#include "test_value.hpp"

using namespace Catch;

namespace cnn::neuron {

SCENARIO("scalar neuron test", "[scalar_neuron]") {

  GIVEN("a source neuron") {
    neuron_mock source(0.2f);
    AND_GIVEN("a scalar neuron based on the given source") {
      scalar_neuron scalar_n(source);
      WHEN("get output from the scalar") {
        auto output = scalar_n.output();
        THEN("its output is same as it source") {
          REQUIRE(output == Approx(source.output()).margin(test_precision));
        }
      }

      AND_GIVEN("a scale") {
        constexpr float scalar = .8f;
        WHEN("it scales with the given scale") {
          scalar_n.scale(scalar);
          THEN("its output is scaled") {
            auto expected = source.output() * scalar;
            REQUIRE(scalar_n.output() ==
                    Approx(expected).margin(test_precision));
          }
        }

        AND_GIVEN("a gradient with") {
          const float gradient = .75f;
          WHEN("it scales") {
            scalar_n.scale(scalar);
            AND_WHEN("it propagates back") {
              scalar_n.propagate_back(gradient);
              THEN("the source has been updated with propagtion value") {
                auto expected = gradient * scalar;
                REQUIRE(source.propagate_back_called_with() ==
                        Approx(expected).margin(test_precision));
              }
            }
          }
        }
      }
    }
  }
}

} // namespace cnn::neuron
#endif // TEST
