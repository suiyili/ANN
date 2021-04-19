#include "relu.hpp"
#include <algorithm>

namespace cnn::neuron {
relu::relu(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
relu::get_result(float z) const noexcept {
  auto output = std::max(0.0f, z);
  auto derivative = z < 0.0f ? 0.0f : 1.0f;
  return {output, derivative};
}
} // namespace cnn::neuron