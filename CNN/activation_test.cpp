#ifdef TEST

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "activation_mock.hpp"

#include "layer_mock.hpp"
#include "test_value.hpp"
#include "value_factory_mock.hpp"
#include <array>
#include <forward_list>
#include <future>
#include <numeric>

using namespace Catch;

namespace cnn::neuron {

SCENARIO("activation test", "[activation]") {

  GIVEN("a previous layer") {
    value_array prev_layer_values{.40f, .25f, .13f, .61f, 1.0f};
    constexpr unsigned short prev_size = 4U;
    auto prev_layer = std::make_unique<layer_mock>(prev_size);
    auto &prev = *prev_layer;
    for (size_t i = 0U; i < prev_size; ++i)
      prev[i].set_output(prev_layer_values[i]);

    AND_GIVEN("a series of W values") {
      value_array init_weights{.20f, .31f, .22f, .85f, .58f};
      value_factory_mock factory(init_weights);
      activation_mock n(std::move(prev_layer), factory);
      WHEN("activation forward") {
        n.forward();
        THEN("it should get argument from linear product") {
          auto expected = factory[prev.size()];
          for (size_t i = 0; i < prev.size(); ++i)
            expected += prev[i].output() * factory[i];

          REQUIRE(n.get_argument() == Approx(expected).margin(test_precision));
        }
      }
      AND_GIVEN("a series of error gredients") {
        std::array<float, 4> error_gradient{.25f, .22f, .38f, .18f};
        WHEN("activation propagates back") {
          n.propagate_back(error_gradient[0]);
          n.train();
          THEN("it should call previous layer with curated value") {
            for (size_t i = 0; i < prev.size(); ++i) {
              auto expected = error_gradient[0] * factory[i];
              REQUIRE(prev[i].propagate_back_called_with() ==
                      Approx(expected).margin(test_precision));
            }

            AND_WHEN("learn from more back propagation") {

              for (size_t i = 1U; i < error_gradient.size(); ++i) {
                n.propagate_back(error_gradient[i]);
                n.train();
              }

              const float learning_rate = .2f;
              n.learn(learning_rate);              

              auto gradient_sum = std::accumulate(error_gradient.begin(),
                                              error_gradient.end(), 0.0f);
              value_array delta = prev_layer_values * gradient_sum;
              delta /= (float)error_gradient.size();
              delta *= learning_rate;
              auto weights = init_weights - delta;

              for (size_t i = 0; i < prev.size(); ++i)
                prev[i].set_output(1.0f);
              n.forward();

              REQUIRE(n.get_argument() ==
                      Approx(weights.sum()).margin(test_precision));
            }
          }
        }

        WHEN("propagate back with multi-threads") {

          std::forward_list<std::future<void>> tasks;

          float total_error = 0.0f;
          for (auto e : error_gradient) {
            tasks.emplace_front(std::async(std::launch::async,
                                           [&n, e] { n.propagate_back(e); }));
            total_error += e;
          }

          for (auto &t : tasks)
            t.wait();
          n.train();
          THEN("it should sum all propagation currectly") {
            REQUIRE(n.get_back_prop() ==
                    Approx(total_error).margin(test_precision));
          }
        }
      }
    }
  }
}

} // namespace cnn::neuron

#endif // TEST