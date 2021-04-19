#pragma once

#include "data_reader.hpp"

class data_cache {
public:
  data_cache(data_reader &reader, size_t count);
  virtual ~data_cache() = 0;
  size_t count() const;

protected:
  static size_t read_count(data_reader &reader, size_t count);

private:
  const size_t count_;
};
