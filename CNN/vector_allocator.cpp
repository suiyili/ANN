#include "vector_allocator.hpp"

namespace cnn {
vector_allocator::vector_allocator(size_t pool_size)
    : buffer_(pool_size), pool_(&buffer_) {}

vector_allocator &get_allocator() {
  static vector_allocator allocator(4096);
  return allocator;
}
} // namespace cnn