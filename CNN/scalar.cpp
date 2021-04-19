#include "scalar.hpp"
#include <stdexcept>

namespace cnn::layer {
void scalar::dropout(float ratio) noexcept {
  if (ratio >= 0.0f && ratio < 1.0f)
    dropout_ratio_ = ratio;
  //else
    //throw std::invalid_argument("invalid rate");
}

scalar::operator float() const noexcept { return 1.0f / (1.0f - dropout_ratio_); }

size_t scalar::scaled(size_t size) const noexcept {
  return (size_t)(size * (1.0f - dropout_ratio_));
}

} // namespace cnn::layer