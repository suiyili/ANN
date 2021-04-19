#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include "input_layer.hpp"

namespace cnn::layer {

SCENARIO("input layer test", "[input_layer]") {
  GIVEN("a input layer with given size") {
    input_layer input({2, 2});
    WHEN("read data") {
      value_array data{.2f, .1f, .3f, .4f};
      input.read(data);
      net_layer_i &input_l = input;
      THEN("it should get data by id") { 
          for (size_t i = 0; i < data.size(); i++) {
          REQUIRE(input_l[i].output() == data[i]);
        }          
      }
    }
  }
}

} // namespace cnn::layer

#endif // TEST