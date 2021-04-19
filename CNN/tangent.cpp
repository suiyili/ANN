#include "tangent.hpp"

namespace cnn::neuron {
tangent::tangent(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
tangent::get_result(const float z) const noexcept {
  auto output = tanh(z);
  auto derivative = (1.0f - output) * (1.0f + output);
  return {output, derivative};
}
} // namespace cnn::neuron