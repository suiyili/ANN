#pragma once
#include "hidden_neuron.hpp"

namespace cnn::neuron {
class tangent final : public hidden_neuron {
public:
  tangent(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  virtual ~tangent() override = default;

  tangent(const tangent &) = delete;
  tangent(tangent &&) = delete;

#ifndef TEST
private:
#endif // !TEST
  virtual activation_result get_result(const float z) const noexcept override;
};
} // namespace cnn::neuron