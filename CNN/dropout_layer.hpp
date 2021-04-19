#pragma once

#include "net_layer_i.hpp"
#include "scalar.hpp"
#include "scalar_neuron.hpp"
#include <vector>
#include <random>
#include <memory_resource>

using namespace cnn::neuron;

namespace cnn::layer {
class dropout_layer final : public net_layer_i {
public:
  dropout_layer(layer_i &prev_layer, const scalar &scale) noexcept;
  ~dropout_layer() noexcept override = default;

#ifndef TEST
private:
#endif // !TEST

  size_t size() const noexcept override;
  simple_neuron_i &operator[](size_t ordinal) noexcept override;
  const tensor_shape &layer_shape() const noexcept override;
  void forward() noexcept override;
  void train() noexcept override;
  void learn(float learning_rate) noexcept override;

  static std::pmr::vector<scalar_neuron> init_scalars(layer_i &prev) noexcept;
  void set_scalars(size_t number_of_scalars) noexcept;

  layer_i &prev_layer_;
  const scalar &scale_;
  std::pmr::vector<scalar_neuron> scalars_;

  std::random_device rd_;
  std::uniform_int_distribution<size_t> dist_;
  std::valarray<bool> scalar_table_;
};
} // namespace cnn::layer