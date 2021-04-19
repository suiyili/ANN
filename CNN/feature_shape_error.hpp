#pragma once

#include <stdexcept>

namespace cnn {
class feature_shape_error final : std::range_error {
public:
  using std::range_error::range_error;
};
} // namespace cnn