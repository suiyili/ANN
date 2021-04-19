#include "image_cache.hpp"

image_cache::image_cache(data_reader &reader, size_t count)
    : data_cache(reader, count), layer_shape_(read_domain(reader)),
      data_(read_image(reader, layer_shape_.volume(), data_cache::count())) {}

const tensor_shape &image_cache::layer_shape() const { return layer_shape_; }

const value_array &image_cache::get_image(size_t ordinal) const {
  return data_[ordinal];
}

tensor_shape image_cache::read_domain(data_reader &reader) {
  const auto first = reader.read_uint();
  const auto second = reader.read_uint();
  return {first, second};
}

std::valarray<value_array> image_cache::read_image(data_reader &reader, size_t size,
                                              size_t count) {
  value_array init_values(size);
  std::valarray<value_array> data(init_values, count);

  byte_array bytes(size);
  for (auto &values : data) {
    reader.read_bytes(bytes);
    for (size_t i = 0; i < bytes.size(); ++i)
      values[i] = (float)bytes[i] / 255U;
  }

  return move(data);
}