#pragma once

#include "dnn_type.hpp"
#include "simple_neuron_i.hpp"

namespace cnn::neuron {
class input_neuron final : public simple_neuron_i {
public:
  input_neuron(const value_array *&source, size_t ordinal) noexcept;
  ~input_neuron() override = default;

  float output() const noexcept override;
  void propagate_back(float error_gradient) noexcept override;

private:
  const value_array *&source_;
  const size_t ordinal_;
};
} // namespace cnn::neuron