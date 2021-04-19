#pragma once

#include "feature_filter.hpp"
#include "layer_factory_i.hpp"
#include "neuron_factory.hpp"

using namespace cnn::layer;

namespace cnn::net {
class feature_layer_factory : public layer_factory_i {
public:
  feature_layer_factory(const tensor_shape &feature_shape,
                        neuron_factory &factory);
  virtual ~feature_layer_factory() = 0;

protected:
  virtual std::unique_ptr<feature_filter>
  create_filter(const tensor_shape &feature_shape,
                const tensor_shape &prev_layer) const = 0;

private:
  virtual std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;
  const tensor_shape feature_shape_;
  neuron_factory &factory_;
};
} // namespace cnn::net