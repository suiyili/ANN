#pragma once

#include "tensor_shape.hpp"

namespace cnn::layer {
class feature_filter {
public:
  feature_filter(const tensor_shape &feature_shape,
                 const tensor_shape &prev_shape) noexcept;
  virtual ~feature_filter() = default;

  neuron_indices operator[](const tensor_vector &pixel) const noexcept;
  virtual const tensor_shape &layer_shape() const = 0;

protected:
  virtual tensor_vector get_feature_start(const tensor_vector &pixel) const = 0;
  const tensor_shape feature_shape_;

private:
  const tensor_shape prev_shape_;
};
} // namespace cnn::layer