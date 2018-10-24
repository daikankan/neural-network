#include <iostream>
#include "binary_data_reader.hpp"

BinaryData::BinaryData(int prefetch_size, int dimension,
                       const char* file_name)
    : PrefetchingData(prefetch_size, dimension)
    , file_in_(file_name, std::ios::binary) {
}

BinaryData::~BinaryData() {
  file_in_.close();
  StopInternalThread();
}

bool BinaryData::load_data(float* data) {
  if (!file_in_.good()) {
    return false;
  }
  int dimension = 0;
  if (file_in_.read((char*)&dimension, sizeof(int))
      && file_in_.read((char*)data, dimension * sizeof(float))) {
    return true;
  } else {
    std::cout << "end of reading" << std::endl;
    return false;
  }
}
