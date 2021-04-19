#include "multi_channel_layer_factory.hpp"
#include <stdexcept>
#include <utility>

namespace cnn::net {
multi_channel_layer_factory::multi_channel_layer_factory(
    tensor_shape unit_domain, size_t channel, neuron_factory_i &factory)
    : unit_domain_(std::move(unit_domain)), channel_(channel), factory_(factory) {}

std::unique_ptr<net_layer_i>
multi_channel_layer_factory::create(layer_i &prev_layer) {
  throw std::runtime_error("not implement");
}
} // namespace cnn::net