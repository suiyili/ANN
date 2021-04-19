#include "dropout_layer_factory.hpp"
#include "dropout_layer.hpp"

using namespace cnn::layer;

namespace cnn::net {
dropout_layer_factory::dropout_layer_factory(const scalar &scale)
    : scale_(scale) {}

std::unique_ptr<net_layer_i> dropout_layer_factory::create(layer_i &prev_layer) {
  return std::make_unique<dropout_layer>(prev_layer, scale_);
}

} // namespace cnn::net