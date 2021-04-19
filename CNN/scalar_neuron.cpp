#include "scalar_neuron.hpp"

namespace cnn::neuron {
scalar_neuron::scalar_neuron(simple_neuron_i &source) noexcept
    : source_(source), scalar_(1.0f) {}

void scalar_neuron::scale(float scalar) noexcept { scalar_ = scalar; }

float scalar_neuron::output() const noexcept {
  return scalar_ * source_.output();
}

void scalar_neuron::propagate_back(float error_gradient) noexcept {
  source_.propagate_back(scalar_ * error_gradient);
}

} // namespace cnn::neuron