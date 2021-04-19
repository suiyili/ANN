#pragma once

#include "layer_i.hpp"

namespace cnn::layer {
class dense_layer_proxy final : public neuron_array_i {
public:
  explicit dense_layer_proxy(neuron_array_i &prev_layer) noexcept;
  ~dense_layer_proxy() override = default;

private:
  simple_neuron_i &operator[](size_t ordinal) noexcept override;
  size_t size() const noexcept override;

  neuron_array_i &prev_layer_;
};
} // namespace cnn::layer