#include "inflation_layer.hpp"
#include "ordinal_cursor.hpp"
#include <stdexcept>

namespace cnn::layer {
inflation_layer::inflation_layer(layer_i &prev_layer,
                                 const tensor_shape &to_inflate)
    : prev_layer_(prev_layer),
      layer_shape_(init_layer_shape(prev_layer_.layer_shape(), to_inflate)),
      indices_(layer_shape_.volume()) {
  auto prev_shape =
      prev_layer.layer_shape().inflate(prev_layer.layer_shape().dimensions());
  ordinal_cursor base_cursor(prev_shape);
  ordinal_cursor unit_cursor(to_inflate);

  const auto base_size = prev_shape.volume();
  const auto unit_size = to_inflate.volume();

  for (size_t i = 0U; i < base_size; ++i) {
    const auto base_pos = base_cursor.increase();
    const auto next_base_pos = base_pos * to_inflate;
    for (size_t j = 0U; j < unit_size; ++j) {
      const auto unit_pos = unit_cursor.increase();
      const auto next_pos = next_base_pos + unit_pos;
      const auto id = layer_shape_.ordinal_of(next_pos);
      indices_[id] = i;
    }
  }
}

simple_neuron_i &inflation_layer::operator[](size_t ordinal) noexcept {
  //if (ordinal < indices_.size())
  return prev_layer_[indices_[ordinal]];
  //throw std::out_of_range("index out of range");
}

size_t inflation_layer::size() const noexcept { return indices_.size(); }

const tensor_shape &inflation_layer::layer_shape() const noexcept {
  return layer_shape_;
}

void inflation_layer::forward() noexcept {}

void inflation_layer::train() noexcept {}

void inflation_layer::learn(float) noexcept {}

tensor_shape inflation_layer::init_layer_shape(const tensor_shape &prev_shape,
                                               const tensor_shape &to_inflate) {
  auto shape = prev_shape.inflate(prev_shape.dimensions());
  if (shape.dimensions() == to_inflate.dimensions())
    return tensor_shape(shape * to_inflate);
  throw std::invalid_argument("different dimensions");
}

} // namespace cnn::layer