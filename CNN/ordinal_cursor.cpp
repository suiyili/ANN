#include "ordinal_cursor.hpp"
#include <stdexcept>

namespace cnn {
ordinal_cursor::ordinal_cursor(
    std::initializer_list<tensor_vector::value_type> range)
    : range_(move(range)), current_(range.size()) {
  if (range_.size() == 0U || range_.min() == 0U) {
    throw std::invalid_argument("every element of range should be greater than 0");
  }
}

ordinal_cursor::ordinal_cursor(const tensor_vector &range) noexcept
    : range_(range), current_(range.size()) {}

const tensor_vector ordinal_cursor::increase() noexcept{
  tensor_vector prev(current_);

  for (size_t i = 0; i < range_.size(); ++i) {
    if (++current_[i] == range_[i]) {
      current_[i] = 0;
      continue;
    } else
      break;
  }

  return prev;
}

} // namespace cnn