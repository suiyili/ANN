#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "dropout_layer.hpp"
#include "layer_mock.hpp"

namespace cnn::layer {

SCENARIO("dropout layer test", "[dropout_layer]") {

  GIVEN("a dropout layer with scalar and prev layer") {
    scalar scalar_v;
    layer_mock prev_layer(20U);
    for (size_t i = 0; i < prev_layer.size(); ++i)
      prev_layer[i].set_output(1.0f);
    dropout_layer dropout_l(prev_layer, scalar_v);
    AND_GIVEN("a dropout ratio between 0 and 1") {
      constexpr float dropout_rate = .25f;
      WHEN("dropout and forward") {
        scalar_v.dropout(dropout_rate);
        dropout_l.learn(.0f);
        THEN("it should drop neurons as the ratio") {
          float expected = dropout_l.size() * dropout_rate;
          size_t dropped = 0;
          for (size_t i = 0; i < dropout_l.size(); ++i)
            if (dropout_l[i].output() == 0.0f)
              ++dropped;
          REQUIRE(dropped == (size_t)expected);
        }
      }
    }
    AND_GIVEN("a zero dropout ratio") {
      constexpr float dropout_rate = 0.f;
      WHEN("dropout and forward") {
        scalar_v.dropout(dropout_rate);
        dropout_l.learn(.0f);
        THEN("it should drop nothing") {
          size_t dropped = 0;
          for (size_t i = 0; i < dropout_l.size(); ++i)
            if (dropout_l[i].output() == 0.0f)
              ++dropped;
          REQUIRE(dropped == 0U);
        }
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST