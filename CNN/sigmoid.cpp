#include "sigmoid.hpp"

namespace cnn::neuron {
sigmoid::sigmoid(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
sigmoid::get_result(const float z) const noexcept {
  auto output = 1.0f / (1.0f + expf(-z));
  auto derivative = output * (1.0f - output);
  return activation_result{output, derivative};
}
} // namespace cnn::neuron