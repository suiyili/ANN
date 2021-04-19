#pragma once

#include <memory_resource>

namespace cnn {

class vector_allocator {
public:
  vector_allocator(size_t pool_size);
  
  template<typename T> std::pmr::vector<T> allocate(size_t capacity) {
    std::pmr::polymorphic_allocator<T> allocator(&pool_);
    std::pmr::vector<T> vec(allocator);
    vec.reserve(capacity);
    return vec;
  }

private:
  std::pmr::monotonic_buffer_resource buffer_;
  std::pmr::unsynchronized_pool_resource pool_;
};

vector_allocator &get_allocator();

} // namespace cnn