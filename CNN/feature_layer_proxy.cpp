#include "feature_layer_proxy.hpp"
#include "vector_allocator.hpp"
#include <stdexcept>

namespace cnn::layer {
feature_layer_proxy::feature_layer_proxy(neuron_array_i &prev_layer,
                                         const neuron_indices &indices) noexcept
    : neurons_(init_neurons(prev_layer, indices)) {}

simple_neuron_i &feature_layer_proxy::operator[](size_t ordinal) noexcept {
  return neurons_[ordinal];
}

size_t feature_layer_proxy::size() const noexcept { return neurons_.size(); }

std::pmr::vector<std::reference_wrapper<simple_neuron_i>>
feature_layer_proxy::init_neurons(neuron_array_i &prev_layer,
                                  const neuron_indices &indices) noexcept {
  auto &allocator = get_allocator();
  auto neurons = allocator.allocate<std::reference_wrapper<simple_neuron_i>>(
      indices.size());
  for (auto &i : indices)
    neurons.emplace_back(prev_layer[i]);
  return neurons;
}
} // namespace cnn::layer