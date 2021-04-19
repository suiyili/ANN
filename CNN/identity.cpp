#include "identity.hpp"

namespace cnn::neuron {
identity::identity(std::unique_ptr<neuron_array_i> inputs,
                   value_factory_i &factory) noexcept
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
identity::get_result(const float z) const noexcept {
  return {z, 1.0f};
}
} // namespace cnn::neuron