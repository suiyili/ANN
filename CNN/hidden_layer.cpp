#include "hidden_layer.hpp"
#include <execution>
#include <stdexcept>

namespace cnn::layer {
hidden_layer::hidden_layer(const tensor_shape &shape,
                           neuron_factory_i &factory) noexcept
    : shape_(shape), neurons_(create_neurons(shape.volume(), factory)) {}

simple_neuron_i &hidden_layer::operator[](size_t ordinal) noexcept {
  // if (ordinal < neurons_.size())
  return *neurons_[ordinal];
  // throw std::out_of_range("out of range");
}

size_t hidden_layer::size() const noexcept { return neurons_.size(); }

const tensor_shape &hidden_layer::layer_shape() const noexcept {
  return shape_;
}

void hidden_layer::forward() noexcept {
  std::for_each(std::execution::par_unseq, std::begin(neurons_),
                std::end(neurons_),
                [](std::unique_ptr<neuron_i> &n) { n->forward(); });
}

void hidden_layer::train() noexcept {
  std::for_each(std::execution::par_unseq, std::begin(neurons_),
                std::end(neurons_),
                [](std::unique_ptr<neuron_i> &n) { n->train(); });
}

void hidden_layer::learn(float learning_rate) noexcept {
  std::for_each(std::execution::par_unseq, std::begin(neurons_),
                std::end(neurons_),
                [&learning_rate](std::unique_ptr<neuron_i> &n) {
                  n->learn(learning_rate);
                });
}

std::valarray<std::unique_ptr<neuron_i>>
hidden_layer::create_neurons(size_t size, neuron_factory_i &factory) noexcept {
  std::valarray<std::unique_ptr<neuron_i>> neurons(size);
  size_t i = 0;
  for (auto &n : neurons)
    n = factory.create();
  return neurons;
}
} // namespace cnn::layer