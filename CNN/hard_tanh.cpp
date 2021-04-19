#include "hard_tanh.hpp"

namespace cnn::neuron {
hard_tanh::hard_tanh(std::unique_ptr<neuron_array_i> inputs,
                     value_factory_i &factory) noexcept
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
hard_tanh::get_result(const float z) const noexcept {
  float output = z > 1.0f ? 1.0f : (z < -1.0f ? -1.0f : z);
  float derivative = (z > 1.0f || z < -1.0f) ? 0.0f : 1.0f;
  return {output, derivative};
}
} // namespace cnn::neuron