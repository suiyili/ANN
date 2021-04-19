#pragma once

#include "data_cache.hpp"

class label_cache final : public data_cache {
public:
  label_cache(data_reader &reader, size_t count);
  virtual ~label_cache() override = default;

  dnn_byte get_label(size_t ordinal) const;

private:
  static byte_array init_labels(data_reader &reader, size_t count);
  const byte_array data_;
};
