#pragma once

#include "simple_neuron_i.hpp"

namespace cnn::neuron {
class scalar_neuron final : public simple_neuron_i {
public:
  explicit scalar_neuron(simple_neuron_i &source) noexcept;
  virtual ~scalar_neuron() override = default;

  void scale(float scalar) noexcept;

  virtual float output() const noexcept override;
  virtual void propagate_back(float error_gradient) noexcept override;

private:
  simple_neuron_i &source_;
  float scalar_;
};
} // namespace cnn::neuron