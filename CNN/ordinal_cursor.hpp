#pragma once

#include "dnn_type.hpp"

namespace cnn {
class ordinal_cursor {
public:
  ordinal_cursor(std::initializer_list<tensor_vector::value_type> range);
  ordinal_cursor(const tensor_vector &range) noexcept;
  const tensor_vector increase() noexcept;

private:
  const tensor_vector range_;
  tensor_vector current_;
};
} // namespace cnn