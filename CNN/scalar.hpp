#pragma once

#include "dnn_type.hpp"

namespace cnn::layer {
class scalar final {
public:
  void dropout(float rate) noexcept;
  explicit operator float() const noexcept;
  [[nodiscard]] size_t scaled(size_t size) const noexcept;

private:
  float dropout_ratio_ = .0f;
};

} // namespace cnn::layer