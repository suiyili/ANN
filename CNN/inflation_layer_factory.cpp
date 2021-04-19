#include "inflation_layer_factory.hpp"
#include "inflation_layer.hpp"

using namespace cnn::layer;

namespace cnn::net {
inflation_layer_factory::inflation_layer_factory(size_t channel)
    : channel_(channel) {}

std::unique_ptr<net_layer_i> inflation_layer_factory::create(layer_i &prev_layer) {
  auto &shape = prev_layer.layer_shape();
  tensor_shape channle_shape(1U, shape.dimensions() + 1U);
  channle_shape[shape.dimensions()] = channel_;
  return make_unique<inflation_layer>(prev_layer, channle_shape);
}
} // namespace cnn::net