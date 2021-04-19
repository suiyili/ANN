#pragma once

#include "hidden_neuron.hpp"

namespace cnn::neuron {
class relu final : public hidden_neuron {
public:
  relu(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  ~relu() override = default;

  relu(const relu &) = delete;
  relu(relu &&) = delete;

#ifndef TEST
private:
#endif // !TEST

  [[nodiscard]] activation_result get_result(float z) const noexcept override;
};
} // namespace cnn::neuron