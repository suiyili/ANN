#pragma once

#include <valarray>
#include <fstream>

using dnn_byte = unsigned char;
using byte_array = std::valarray<dnn_byte>;

class data_reader final {
public:
  data_reader(const std::string &file);
  ~data_reader();

  size_t read_uint();
  void read_bytes(byte_array &buffer);

private:
  std::ifstream reader_;
};