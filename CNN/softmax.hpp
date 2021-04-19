#pragma once

#include "simple_neuron_i.hpp"

namespace cnn::neuron {
class softmax final : public simple_neuron_i {
public:
  explicit softmax(simple_neuron_i &input) noexcept;
  ~softmax() override = default;

  float output() const noexcept override;
  void set_output(float output) noexcept;
  void forward_output() noexcept;

#ifndef TEST
private:
#endif // TEST

  void propagate_back(float error_gradient) noexcept override;

  simple_neuron_i &input_;
  float output_;
};
} // namespace cnn::neuron