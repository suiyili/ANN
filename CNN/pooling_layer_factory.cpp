#include "pooling_layer_factory.hpp"
#include "pooling_filter.hpp"
#include "pooling_neuron_factory.hpp"

namespace cnn::net {
pooling_layer_factory::pooling_layer_factory(const tensor_shape &feature_shape,
                                             neuron_factory &factory)
    : feature_layer_factory(feature_shape, factory) {}

std::unique_ptr<feature_filter>
pooling_layer_factory::create_filter(const tensor_shape &feature_shape,
                                     const tensor_shape &prev_layer) const {
  return make_unique<pooling_filter>(feature_shape, prev_layer);
}

} // namespace cnn::net