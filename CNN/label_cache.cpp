#include "label_cache.hpp"

label_cache::label_cache(data_reader &reader, size_t count)
    : data_cache(reader, count),
      data_(init_labels(reader, data_cache::count())) {}

dnn_byte label_cache::get_label(size_t ordinal) const { return data_[ordinal]; }

byte_array label_cache::init_labels(data_reader &reader, size_t count) {
  byte_array labels(count);
  reader.read_bytes(labels);
  return move(labels);
}
