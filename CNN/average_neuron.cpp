#include "average_neuron.hpp"

namespace cnn::neuron {
average_neuron::average_neuron(std::unique_ptr<neuron_array_i> inputs) noexcept
    : inputs_(move(inputs)), average_(0.f) {}

void average_neuron::forward() noexcept {
  auto &inputs = *inputs_;
  average_ = 0.f;
  for (size_t i = 0; i < inputs.size(); ++i) {
    decltype(auto) n = inputs[i];
    average_ += n.output();
  }
  average_ /= inputs.size();
}

float average_neuron::output() const noexcept { return average_; }

void average_neuron::propagate_back(float error_gradient) noexcept {
  auto &inputs = *inputs_;
  error_gradient /= inputs.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    decltype(auto) n = inputs[i];
    n.propagate_back(error_gradient);
  }
}

void average_neuron::train() noexcept {}

void average_neuron::learn(float learning_rate) noexcept {}
} // namespace cnn::neuron
