#pragma once

#include "neuron_array_i.hpp"
#include "neuron_i.hpp"
#include "value_factory_i.hpp"
#include <atomic>
#include <memory>

namespace cnn::neuron {
class activation : public neuron_i {
public:
  activation(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  ~activation() = 0;

  activation(const activation &) = delete;
  activation(activation &&) = delete;

protected:
  virtual void set_argument(float z) = 0;
  virtual float get_error_gradient(float back_prop) = 0;

#ifdef TEST
public:
#else
private:
#endif // TEST

  void forward() noexcept final;
  void propagate_back(float error_gradient) noexcept final;
  void train() noexcept final;
  void learn(float learning_rate) noexcept final;

  std::unique_ptr<neuron_array_i> inputs_;

  value_array weights_;
  value_array delta_;

  std::atomic<float> back_error_gradient_;

  size_t count_;
};
} // namespace cnn::neuron