#include "hidden_neuron.hpp"

namespace cnn::neuron {
hidden_neuron::hidden_neuron(std::unique_ptr<neuron_array_i> inputs,
                             value_factory_i &factory) noexcept
    : activation(move(inputs), factory), result_({0, 0}) {}

float hidden_neuron::output() const noexcept { return result_.output; }

void hidden_neuron::set_argument(const float z) noexcept {
  result_ = get_result(z);
}

float hidden_neuron::get_error_gradient(const float back_prop) noexcept {
  return back_prop * result_.derivative;
}
} // namespace cnn::neuron