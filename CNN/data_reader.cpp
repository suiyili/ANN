#include "data_reader.hpp"
#include <bitset>
#include <stdexcept>

data_reader::data_reader(const std::string &file)
    : reader_(file, std::ios::binary) {
  if (reader_.bad())
    throw std::invalid_argument("can't open file");
}

data_reader::~data_reader() { reader_.close(); }

size_t data_reader::read_uint() {
  std::bitset<32> value;
  dnn_byte buffer = 0x00;
  for (size_t i = 0; i < 4; ++i) {
    reader_.read((char *)&buffer, 1U);
    value <<= 8;
    value |= buffer;
  }
  return value.to_ulong();
}

void data_reader::read_bytes(byte_array &buffer) {
  reader_.read((char *)begin(buffer), buffer.size());
}