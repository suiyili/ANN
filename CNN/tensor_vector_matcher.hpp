#pragma once

#include <catch2/matchers/catch_matchers.hpp>
#include "dnn_type.hpp"

using namespace cnn;
using namespace Catch::Matchers;

class tensor_vector_matcher final : public MatcherBase<tensor_vector> {
public:
  explicit tensor_vector_matcher(const tensor_vector &expected);
  ~tensor_vector_matcher() override = default;

  std::string describe() const override;
  bool match(const tensor_vector &actual) const override;

private:
  const tensor_vector &expected_;
};

tensor_vector_matcher Equals(const tensor_vector &expected);
