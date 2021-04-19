#include "softmax.hpp"

namespace cnn::neuron {
softmax::softmax(simple_neuron_i &input) noexcept
    : input_(input), output_(0.0f) {}

float softmax::output() const noexcept { return output_; }

void softmax::set_output(float output) noexcept { output_ = output; }

void softmax::forward_output() noexcept { output_ = input_.output(); }

void softmax::propagate_back(float error_gradient) noexcept {
  return input_.propagate_back(output_ - error_gradient);
}

} // namespace cnn::neuron