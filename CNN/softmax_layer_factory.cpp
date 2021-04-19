#include "softmax_layer_factory.hpp"
#include "softmax_layer.hpp"

using namespace cnn::layer;

namespace cnn::net {
std::unique_ptr<net_layer_i> softmax_layer_factory::create(layer_i &prev_layer) {
  return std::make_unique<softmax_layer>(prev_layer);
}
} // namespace cnn::net
