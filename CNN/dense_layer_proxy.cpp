#include "dense_layer_proxy.hpp"

namespace cnn::layer {
dense_layer_proxy::dense_layer_proxy(neuron_array_i &prev_layer) noexcept
    : prev_layer_(prev_layer) {}

simple_neuron_i &dense_layer_proxy::operator[](size_t ordinal) noexcept {
  return prev_layer_[ordinal];
}

size_t dense_layer_proxy::size() const noexcept { return prev_layer_.size(); }
} // namespace cnn::layer