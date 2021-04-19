#include "feature_filter.hpp"
#include "ordinal_cursor.hpp"

namespace cnn::layer {
feature_filter::feature_filter(const tensor_shape &feature_shape,
                               const tensor_shape &prev_shape) noexcept
    : feature_shape_(feature_shape), prev_shape_(prev_shape) {}


neuron_indices feature_filter::operator[](const tensor_vector &pixel) const noexcept {
  neuron_indices feature_indices(feature_shape_.volume());
  ordinal_cursor feature_counter(feature_shape_);
  auto feature_start = get_feature_start(pixel);
  for (size_t j = 0; j < feature_indices.size(); ++j) {
    auto feature_pos = feature_counter.increase();

    auto net_pos = feature_start + feature_pos;
    auto net_ordinal = prev_shape_.ordinal_of(net_pos);

    feature_indices[j] = net_ordinal;
  }
  return feature_indices;
}
} // namespace cnn::layer