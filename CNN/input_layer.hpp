#pragma once

#include "input_neuron.hpp"
#include "net_layer_i.hpp"
#include <vector>
#include <memory_resource>

using namespace cnn::neuron;

namespace cnn::layer {
class input_layer final : public net_layer_i {
public:
  explicit input_layer(const tensor_shape &layer_shape) noexcept;
  ~input_layer() override = default;

  void read(const value_array &data) noexcept;

private:
  input_neuron &operator[](size_t ordinal) noexcept override;
  const tensor_shape &layer_shape() const noexcept override;
  size_t size() const  noexcept override;

  void train() noexcept override;
  void forward() noexcept override;
  void learn(float learning_rate) noexcept override;

  static std::pmr::vector<input_neuron> init_neurons(const value_array *&source,
                                           size_t size) noexcept;
  const tensor_shape layer_shape_;
  const value_array *source_;
  std::pmr::vector<input_neuron> neurons_;
};
} // namespace cnn::layer