#include "activation.hpp"

namespace cnn::neuron {
activation::activation(std::unique_ptr<neuron_array_i> inputs,
                       value_factory_i &factory) noexcept
    : inputs_(move(inputs)), weights_(factory.generate(inputs_->size() + 1U)),
      delta_(0.0f, weights_.size()), back_error_gradient_(0.0f), count_(0U) {}

activation::~activation() {}

void activation::forward() noexcept {
  auto &inputs = *inputs_;
  float z = weights_[inputs.size()];
  for (size_t i = 0; i < inputs.size(); ++i)
    z += weights_[i] * inputs[i].output();
  set_argument(z);
}

void activation::propagate_back(float error_gradient) noexcept {
  float f = back_error_gradient_.load(std::memory_order_relaxed);
  float desired = f + error_gradient;
  while (!back_error_gradient_.compare_exchange_weak(f, desired,
                                  std::memory_order_release,
                                  std::memory_order_consume))
    desired = back_error_gradient_ + f;
}

void activation::train() noexcept {
  auto back_error = back_error_gradient_.load(std::memory_order_relaxed);
  const auto error_gradient = get_error_gradient(back_error);
  auto &inputs = *inputs_;

  for (size_t i = 0; i < inputs.size(); ++i) {
    delta_[i] += error_gradient * inputs[i].output();
    inputs[i].propagate_back(weights_[i] * error_gradient);
  }
  delta_[inputs.size()] += error_gradient;

  back_error_gradient_.store(0.0f, std::memory_order_release);
  ++count_;
}

void activation::learn(float learning_rate) noexcept {
  if(count_ > 0){
    delta_ /= (float)count_;
    delta_ *= learning_rate;
    weights_ -= delta_;
  }

  delta_ = .0f;
  count_ = 0U;
}

} // namespace cnn::neuron