#include "feature_layer_factory.hpp"
#include "feature_layer_proxy.hpp"
#include "hidden_layer.hpp"
#include "ordinal_cursor.hpp"

namespace cnn::net {
feature_layer_factory::feature_layer_factory(const tensor_shape &feature_shape,
                                             neuron_factory &factory)
    : feature_shape_(feature_shape), factory_(factory) {}

feature_layer_factory::~feature_layer_factory() = default;

std::unique_ptr<net_layer_i> feature_layer_factory::create(layer_i &prev_layer) {
  auto fil = create_filter(feature_shape_, prev_layer.layer_shape());
  auto &filter = *fil;

  decltype(auto) next_shape = filter.layer_shape();
  auto size = next_shape.volume();

  ordinal_cursor counter(next_shape);

  for (size_t i = 0; i < size; ++i) {
    auto feature_indices = filter[counter.increase()];
    factory_.append(
        make_unique<feature_layer_proxy>(prev_layer, feature_indices));
  }

  return make_unique<hidden_layer>(next_shape.shrink(), factory_);
}
} // namespace cnn::net