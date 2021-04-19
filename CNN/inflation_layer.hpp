#pragma once

#include "net_layer_i.hpp"

namespace cnn::layer {
class inflation_layer final : public net_layer_i {
public:
  inflation_layer(layer_i &prev_layer, const tensor_shape &to_inflate);
  virtual ~inflation_layer() noexcept override = default;

  virtual simple_neuron_i &operator[](size_t ordinal) noexcept override;
  virtual size_t size() const noexcept override;
  virtual const tensor_shape &layer_shape() const noexcept override;
  virtual void forward() noexcept override;
  virtual void train() noexcept override;
  virtual void learn(float learning_rate) noexcept override;

private:
  static tensor_shape init_layer_shape(const tensor_shape &prev_shape,
                                       const tensor_shape &to_inflate);
  layer_i &prev_layer_;
  const tensor_shape layer_shape_;
  neuron_indices indices_;
};
} // namespace cnn::layer