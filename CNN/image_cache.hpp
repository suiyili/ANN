#pragma once

#include "data_cache.hpp"
#include "dnn_type.hpp"
#include "tensor_shape.hpp"

using namespace cnn;

class image_cache final : public data_cache {
public:
  image_cache(data_reader &reader, size_t count);
  ~image_cache() override = default;

  const tensor_shape &layer_shape() const;
  const value_array &get_image(size_t ordinal) const;

private:
  static tensor_shape read_domain(data_reader &reader);
  static std::valarray<value_array> read_image(data_reader &reader, size_t size,
                                          size_t count);

  const tensor_shape layer_shape_;
  std::valarray<value_array> data_;
};