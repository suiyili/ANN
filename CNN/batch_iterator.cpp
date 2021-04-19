#include "batch_iterator.hpp"
#include <stdexcept>

namespace cnn {
batch_iterator::batch_iterator(const data_indices &source, size_t size)
    : source_(source), size_(size), cursor_(0U) {
  if (source.size() % size > 0U)
    throw std::invalid_argument("given source does not match batch size");
}

size_t batch_iterator::size() const noexcept { return size_; }

size_t batch_iterator::operator[](size_t ordinal) noexcept {
  //if (ordinal < size_) {
    return source_[cursor_ + ordinal];
  //}
  //throw std::out_of_range("ordinal out of batch size");
}

bool batch_iterator::move_next() {
  const auto next = cursor_ + size_;
  auto move = next < source_.size();
  if (move)
    cursor_ = next;
  return move;
}
} // namespace cnn
