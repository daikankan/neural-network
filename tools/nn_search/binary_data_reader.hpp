#ifndef _BINARY_DATA_READER_
#define _BINARY_DATA_READER_

#include <fstream>
#include "prefetching_data.hpp"

class BinaryData : public PrefetchingData {
 public:
  explicit BinaryData(int prefetch_size, int dimension,
                      const char* file_name);
  virtual ~BinaryData();

 public:
  virtual bool load_data(float* data);

 private:
  std::ifstream file_in_;
};

#endif
