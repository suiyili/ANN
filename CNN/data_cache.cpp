#include "data_cache.hpp"

data_cache::data_cache(data_reader &reader, size_t count)
    : count_(read_count(reader, count)) {}

data_cache::~data_cache() {}

size_t data_cache::count() const { return count_; }

size_t data_cache::read_count(data_reader &reader, size_t count) {
  if (count == 0U)
    throw std::invalid_argument("invalid count");
  const auto magic_number = reader.read_uint();
  const auto total = reader.read_uint();
  return count > total ? total : count;
}