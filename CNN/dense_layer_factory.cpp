#include "dense_layer_factory.hpp"
#include "dense_layer_proxy.hpp"
#include "hidden_layer.hpp"

using namespace cnn::layer;

namespace cnn::net {
dense_layer_factory::dense_layer_factory(const tensor_shape &layer_shape,
                                         neuron_factory &factory)
    : layer_shape_(layer_shape), factory_(factory) {}

std::unique_ptr<net_layer_i> dense_layer_factory::create(layer_i &prev_layer) {
  auto size = layer_shape_.volume();
  for (size_t i = 0U; i < size; ++i)
    factory_.append(std::make_unique<dense_layer_proxy>(prev_layer));
  return make_unique<hidden_layer>(layer_shape_, factory_);
}
} // namespace cnn::net