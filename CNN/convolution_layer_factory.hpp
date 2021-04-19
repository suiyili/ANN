#pragma once

#include "activation_type.hpp"
#include "feature_layer_factory.hpp"
#include "neuron_factory_i.hpp"

namespace cnn::net {
class convolution_layer_factory final : public feature_layer_factory {
public:
  convolution_layer_factory(const tensor_shape &feature_shape,
                            neuron_factory &factory);
  virtual ~convolution_layer_factory() override = default;

private:
  virtual std::unique_ptr<feature_filter>
  create_filter(const tensor_shape &feature_shape,
                const tensor_shape &prev_layer) const override;
};
} // namespace cnn::net