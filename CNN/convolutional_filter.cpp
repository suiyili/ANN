#include "convolutional_filter.hpp"
#include "feature_shape_error.hpp"

namespace cnn::layer {
convolutional_filter::convolutional_filter(const tensor_shape &feature_shape,
                                           const tensor_shape &prev_shape)
    : feature_filter(feature_shape, prev_shape),
      layer_shape_(init_layer_shape(feature_shape, prev_shape)) {}

const tensor_shape &convolutional_filter::layer_shape() const noexcept{
  return layer_shape_;
}

tensor_vector
convolutional_filter::get_feature_start(const tensor_vector &pixel) const{
  if (layer_shape_.includes(pixel))
    return pixel;
  throw std::out_of_range("the pixel is out of range");
}

tensor_shape
convolutional_filter::init_layer_shape(const tensor_shape &feature_shape,
                                       const tensor_shape &prev_shape) {
  if (prev_shape.dimensions() != feature_shape.dimensions())
    throw feature_shape_error("shape range(size) is not equal");

  tensor_shape layer_shape(prev_shape.dimensions());
  for (size_t i = 0; i < feature_shape.dimensions(); ++i) {
    if (feature_shape[i] > prev_shape[i])
      throw feature_shape_error("given size is out of range.");
    layer_shape[i] = (size_t)(prev_shape[i] - feature_shape[i] + 1);
  }
  return layer_shape;
}
} // namespace cnn::layer