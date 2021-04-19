#pragma once

#include <valarray>

namespace cnn {
using data_indices = std::valarray<size_t>;

class batch_iterator {
public:
  batch_iterator(const data_indices &source, size_t size);
  batch_iterator(const batch_iterator &) = delete;
  batch_iterator &operator=(const batch_iterator &) = delete;

  size_t size() const noexcept;
  size_t operator[](size_t ordinal) noexcept;

  bool move_next();

private:
  const data_indices &source_;
  const size_t size_;
  size_t cursor_;
};
} // namespace cnn