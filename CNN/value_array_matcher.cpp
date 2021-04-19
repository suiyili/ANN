#include "value_array_matcher.hpp"
#include "test_value.hpp"
#include <catch2/catch_approx.hpp>

using namespace Catch;

value_array_matcher::value_array_matcher(const value_array &expected)
    : expected_(expected) {}

std::string value_array_matcher::describe() const { return "compare value array"; }

bool value_array_matcher::match(const value_array &actual) const {
  auto target = Approx(0).margin(test_precision);
  auto comp = abs(expected_ - actual);
  auto re = comp.max() == target;
  return re;
}

value_array_matcher Equals(const value_array &expected) {
  return value_array_matcher(expected);
}
