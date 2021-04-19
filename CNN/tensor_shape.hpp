#pragma once

#include "dnn_type.hpp"

namespace cnn {
class tensor_shape final : public tensor_vector {
public:
  using tensor_vector::tensor_vector;
  explicit tensor_shape(tensor_vector &&other) noexcept;
  size_t volume() const noexcept;
  size_t dimensions() const noexcept;
  size_t ordinal_of(const tensor_vector &pixel) const;
  bool includes(const tensor_vector &pixel) const noexcept;
  tensor_shape shrink() const noexcept;
  tensor_shape
  inflate(size_t ordinal = std::numeric_limits<size_t>::max()) const noexcept;
};
} // namespace cnn