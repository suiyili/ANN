#pragma once

#include "neuron_array_i.hpp"
#include <vector>
#include <memory_resource>

namespace cnn::layer {
class feature_layer_proxy final : public neuron_array_i {
public:
  feature_layer_proxy(neuron_array_i &prev_layer,
                      const neuron_indices &indices) noexcept;
  ~feature_layer_proxy() override = default;

  simple_neuron_i &operator[](size_t ordinal) noexcept override;
  size_t size() const noexcept override;

private:
  std::pmr::vector<std::reference_wrapper<simple_neuron_i>>
  init_neurons(neuron_array_i &prev_layer,
               const neuron_indices &indices) noexcept;

  std::pmr::vector<std::reference_wrapper<simple_neuron_i>> neurons_;
};

} // namespace cnn::layer