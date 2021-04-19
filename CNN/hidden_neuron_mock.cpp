#include "hidden_neuron_mock.hpp"

hidden_neuron_mock::hidden_neuron_mock(std::unique_ptr<neuron_array_i> inputs,
                                       value_factory_i &factory)
    : hidden_neuron(move(inputs), factory) {}

typename hidden_neuron::activation_result
hidden_neuron_mock::get_result(const float z) const {
  z_ = z;
  return {output_, derivative_};
}

void hidden_neuron_mock::set_output(float output) { output_ = output; }

void hidden_neuron_mock::set_derivative(float derivative) {
  derivative_ = derivative;
}

float hidden_neuron_mock::call_get_result_with() const { return z_; }