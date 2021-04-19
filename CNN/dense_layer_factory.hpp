#pragma once

#include "layer_factory_i.hpp"
#include "neuron_factory.hpp"

namespace cnn::net {
class dense_layer_factory final : public layer_factory_i {
public:
  dense_layer_factory(const tensor_shape &layer_shape, neuron_factory &factory);
  ~dense_layer_factory() override = default;
  std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override final;

private:
  const tensor_shape layer_shape_;
  neuron_factory &factory_;
};
} // namespace cnn::net