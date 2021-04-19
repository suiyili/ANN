#pragma once
#include "value_factory_i.hpp"

using namespace cnn;

class value_factory_mock final : public value_factory_i {
public:
  explicit value_factory_mock(std::initializer_list<float> values);
  explicit value_factory_mock(const value_array &values);
  value_factory_mock(float init, size_t size);
  ~value_factory_mock() override = default;

  float operator[](size_t ordinal);

private:
  value_array generate(size_t size) override;
  const value_array values_;
  mutable size_t cursor_;
};
