#pragma once

#include "dnn_type.hpp"
#include <catch2/matchers/catch_matchers.hpp>

using namespace cnn;
using namespace Catch::Matchers;

class value_array_matcher final : public MatcherBase<value_array> {
public:
  explicit value_array_matcher(const value_array &expected);
  ~value_array_matcher() override = default;

  std::string describe() const override;
  bool match(const value_array &actual) const override;

private:
  const value_array &expected_;
};

value_array_matcher Equals(const value_array &expected);