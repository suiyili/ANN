#include "tensor_shape.hpp"
#include <numeric>
#include <stdexcept>

namespace cnn {
tensor_shape::tensor_shape(tensor_vector &&other) noexcept
    : tensor_vector(move(other)) {}

size_t tensor_shape::volume() const noexcept {
  return std::accumulate<decltype(begin(*this)), tensor_vector::value_type>(
      begin(*this), end(*this), 1U, std::multiplies<>());
}

size_t tensor_shape::dimensions() const noexcept { return size(); }

size_t tensor_shape::ordinal_of(const tensor_vector &pixel) const {
  const auto len = size();

  if (pixel.size() != len)
    throw std::invalid_argument("point dimensions are different");
  if (!includes(pixel))
    throw std::range_error("not within tensor_shape");

  size_t ordinal = pixel[0];

  for (size_t i = 1U; i < len; ++i) {
    auto sub = pixel[i];
    sub *= std::accumulate<decltype(begin(*this)), tensor_vector::value_type>(
        begin(*this), begin(*this) + i, 1U, std::multiplies<>());
    ordinal += sub;
  }

  return ordinal;
}

bool tensor_shape::includes(const tensor_vector &pixel) const noexcept{
  auto comp = pixel < (*this);
  return comp.min();
}

tensor_shape tensor_shape::shrink() const noexcept{
  const size_t value = 1U;
  decltype(auto) shape = *this;
  return tensor_shape{shape[shape > value]};
}

tensor_shape tensor_shape::inflate(size_t ordinal) const noexcept{
  const size_t value = 1U;
  decltype(auto) shape = *this;

  if (ordinal > shape.size())
    ordinal = shape.size();
  tensor_vector inflated(shape.size() + 1U);
  for (size_t i = 0U; i < ordinal; ++i)
    inflated[i] = shape[i];
  inflated[ordinal] = value;
  for (size_t i = ordinal; i < shape.size(); ++i)
    inflated[i + 1U] = shape[i];
  return tensor_shape{move(inflated)};
}
} // namespace cnn