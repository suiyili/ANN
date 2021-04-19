#include "convolution_layer_factory.hpp"
#include "activation_factory.hpp"
#include "convolutional_filter.hpp"
#include "feature_shape_error.hpp"

namespace cnn::net {
convolution_layer_factory::convolution_layer_factory(
    const tensor_shape &feature_shape, neuron_factory &factory)
    : feature_layer_factory(feature_shape, factory) {}

std::unique_ptr<feature_filter>
convolution_layer_factory::create_filter(const tensor_shape &feature_shape,
                                         const tensor_shape &prev_layer) const {
  return make_unique<convolutional_filter>(feature_shape, prev_layer);
}

} // namespace cnn::net