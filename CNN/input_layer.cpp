#include "input_layer.hpp"
#include "vector_allocator.hpp"
#include <stdexcept>

namespace cnn::layer {
input_layer::input_layer(const tensor_shape &layer_shape) noexcept
    : layer_shape_(layer_shape), source_(nullptr),
      neurons_(init_neurons(source_, layer_shape.volume())) {}

void input_layer::read(const value_array &data) noexcept {
  /*if (neurons_.size() > data.size())
    throw std::length_error("source is too small");*/
  source_ = &data;
}

void input_layer::forward() noexcept {}

void input_layer::learn(float learning_rate) noexcept {}

void input_layer::train() noexcept {}

input_neuron &input_layer::operator[](size_t ordinal) noexcept {
  // if (ordinal < neurons_.size())
  return neurons_[ordinal];
  // throw std::out_of_range("out of size range");
}

const tensor_shape &input_layer::layer_shape() const noexcept {
  return layer_shape_;
}

size_t input_layer::size() const noexcept { return neurons_.size(); }

std::pmr::vector<input_neuron>
input_layer::init_neurons(const value_array *&source, size_t size) noexcept {
  auto &allocator = get_allocator();
  auto neurons = allocator.allocate<input_neuron>(size);

  for (size_t i = 0U; i < size; ++i)
    neurons.emplace_back(source, i);

  return neurons;
}

} // namespace cnn::layer