#include "max_neuron.hpp"

namespace cnn::neuron {
max_neuron::max_neuron(std::unique_ptr<neuron_array_i> inputs) noexcept
    : inputs_(move(inputs)), max_(&(*inputs_)[0]) {}

void max_neuron::forward() noexcept {
  auto &inputs = *inputs_;
  for (size_t i = 0; i < inputs.size(); ++i) {
    decltype(auto) n = inputs[i];
    if (max_->output() < n.output())
      max_ = &n;
  }
}

float max_neuron::output() const noexcept { return max_->output(); }

void max_neuron::propagate_back(float error_gradient) noexcept {
  max_->propagate_back(error_gradient);
}

void max_neuron::train() noexcept {}

void max_neuron::learn(float learning_rate) noexcept {}
} // namespace cnn::neuron