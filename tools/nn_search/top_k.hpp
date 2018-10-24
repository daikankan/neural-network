#ifndef _TOP_K_
#define _TOP_K_

#include <vector>
#include <boost/shared_ptr.hpp>
#include "binary_data_reader.hpp"

using std::vector;
using boost::shared_ptr;

struct ITEM {
  int index;
  float score;
};

class TopK {
 public:
  explicit TopK(int prefetch_size, int dimension, const char* file_name);
  ~TopK();

 public:
  void get_top_k(float* query, int k, int* indices);

 private:
  void create_heap(int k);
  void sort_heap(int k, int p);

 private:
  shared_ptr<PrefetchingData> prefetch_data_;
  int dimension_;
  ITEM item_[50];
};

#endif
