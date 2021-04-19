#include "activation_mock.hpp"
#include <stdexcept>

activation_mock::activation_mock(std::unique_ptr<neuron_array_i> inputs,
                                 value_factory_i &factory)
    : activation(move(inputs), factory) {}

float activation_mock::get_argument() const { return z_; }

float activation_mock::get_back_prop() const { return prop_; }

float activation_mock::output() const { throw std::logic_error("not support"); }

void activation_mock::set_argument(const float z) { z_ = z; }

float activation_mock::get_error_gradient(const float back_prop) {
  prop_ = back_prop;
  return back_prop;
}
