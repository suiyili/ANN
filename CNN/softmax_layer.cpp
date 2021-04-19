#include "softmax_layer.hpp"
#include "vector_allocator.hpp"
#include <stdexcept>

namespace cnn::layer {
softmax_layer::softmax_layer(layer_i &prev_layer) noexcept
    : prev_layer_(prev_layer), neurons_(init_neurons(prev_layer)) {}

size_t softmax_layer::size() const noexcept { return neurons_.size(); }

simple_neuron_i &softmax_layer::operator[](size_t ordinal) noexcept {
  // if (ordinal < neurons_.size())
  return neurons_[ordinal];
  // throw std::out_of_range("out of range");
}

const tensor_shape &softmax_layer::layer_shape() const noexcept {
  return prev_layer_.layer_shape();
}

void softmax_layer::forward() noexcept {
  for (auto &n : neurons_)
    n.forward_output();

  update(neurons_);
}

void softmax_layer::train() noexcept {}

void softmax_layer::learn(float) noexcept {}

void softmax_layer::update(
    std::pmr::vector<softmax> &neurons) noexcept {
  value_array outputs(neurons.size());
  for (size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = neurons[i].output();

  optimize(outputs);
  set_output(outputs);

  for (size_t i = 0; i < outputs.size(); ++i)
    neurons[i].set_output(outputs[i]);
}

std::pmr::vector<softmax>
softmax_layer::init_neurons(neuron_array_i &neurons) noexcept {
  auto &allocator = get_allocator();
  auto outputs = allocator.allocate<softmax>(neurons.size());
  for (size_t i = 0; i < neurons.size(); ++i)
    outputs.emplace_back(neurons[i]);
  return outputs;
}

void softmax_layer::optimize(value_array &z_values) noexcept {
  auto mean = z_values.sum() / z_values.size();
  z_values -= mean;
}

void softmax_layer::set_output(value_array &outputs) noexcept {
  for (auto &z : outputs)
    z = exp(z);
  auto total = outputs.sum();
  for (auto &z : outputs)
    z /= total;
}
} // namespace cnn::layer