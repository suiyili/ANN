#pragma once

#include "activation.hpp"

namespace cnn::neuron {
class hidden_neuron : public activation {
public:
  hidden_neuron(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  ~hidden_neuron() override = default;

  [[nodiscard]] float output() const noexcept final;

protected:
  struct activation_result {
    float output;
    float derivative;
  };
  [[nodiscard]] virtual activation_result get_result(float z) const = 0;

#ifdef TEST
public:
#else
private:
#endif // TEST
  void set_argument(float z) noexcept final;
  float get_error_gradient(float back_prop) noexcept final;

  activation_result result_;
};
} // namespace cnn::neuron