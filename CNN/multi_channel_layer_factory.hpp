#pragma once

#include "layer_factory_i.hpp"
#include "neuron_factory.hpp"

namespace cnn::net {
class multi_channel_layer_factory final : public layer_factory_i {
public:
  multi_channel_layer_factory(tensor_shape unit_domain, size_t channel,
                              neuron_factory_i &factory);
  ~multi_channel_layer_factory() override = default;
#ifdef TEST
public:
#else
private:
#endif // TEST
  std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;
  const tensor_shape unit_domain_;
  const size_t channel_;
  neuron_factory_i &factory_;
};
} // namespace cnn::net