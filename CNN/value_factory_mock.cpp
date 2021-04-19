#include "value_factory_mock.hpp"
#include <stdexcept>

value_factory_mock::value_factory_mock(std::initializer_list<float> values)
    : values_(move(values)), cursor_(0) {}

value_factory_mock::value_factory_mock(const value_array &values)
    : values_(values) {}

value_factory_mock::value_factory_mock(float init, size_t size)
    : values_(init, size), cursor_(0) {}

float value_factory_mock::operator[](size_t ordinal) {
  return values_[ordinal];
}

value_array value_factory_mock::generate(size_t size) {
  if (values_.size() < size)
    throw std::out_of_range("no more value!");
  return values_[std::slice(0U, size, 1U)];
}
