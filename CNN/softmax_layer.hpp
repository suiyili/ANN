#pragma once

#include "softmax.hpp"
#include "net_layer_i.hpp"
#include <vector>
#include <memory_resource>

using namespace cnn::neuron;

namespace cnn::layer {
class softmax_layer final : public net_layer_i {
public:
  explicit softmax_layer(layer_i &prev_layer) noexcept;
  ~softmax_layer() override = default;

#ifndef TEST
private:
#endif // !TEST

  [[nodiscard]] size_t size() const noexcept override;
  simple_neuron_i &operator[](size_t ordinal) noexcept final;
  [[nodiscard]] const tensor_shape &layer_shape() const noexcept final;

  void forward() noexcept override;
  void train() noexcept final;
  void learn(float learning_rate) noexcept final;

  static void update(std::pmr::vector<softmax> &neurons) noexcept;
  static std::pmr::vector<softmax> init_neurons(neuron_array_i &neurons) noexcept;
  static void optimize(value_array &z_values) noexcept;
  static void set_output(value_array &outputs) noexcept;

  layer_i &prev_layer_;
  std::pmr::vector<softmax> neurons_;
};
} // namespace cnn::layer