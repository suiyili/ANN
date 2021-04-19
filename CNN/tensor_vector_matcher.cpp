#include "tensor_vector_matcher.hpp"

tensor_vector_matcher::tensor_vector_matcher(const tensor_vector &expected)
    : expected_(expected) {}

std::string tensor_vector_matcher::describe() const { return "compare dnn vector"; }

bool tensor_vector_matcher::match(const tensor_vector &actual) const {
  return actual.size() == expected_.size() && (actual == expected_).min();
}

tensor_vector_matcher Equals(const tensor_vector &expected) {
  return tensor_vector_matcher(expected);
}
