#pragma once

#include "feature_filter.hpp"

namespace cnn::layer {
class convolutional_filter final : public feature_filter {
public:
  convolutional_filter(const tensor_shape &feature_shape,
                       const tensor_shape &prev_shape);
  ~convolutional_filter() override = default;
  const tensor_shape &layer_shape() const noexcept override;

private:
  tensor_vector get_feature_start(const tensor_vector &pixel) const override;

  tensor_shape init_layer_shape(const tensor_shape &feature_shape,
                                const tensor_shape &prev_shape);
  const tensor_shape layer_shape_;
};
} // namespace cnn::layer