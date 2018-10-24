#include "top_k.hpp"

TopK::TopK(int prefetch_size, int dimension, const char* file_name)
    : prefetch_data_(new BinaryData(prefetch_size, dimension, file_name))
    , dimension_(dimension) {
}

TopK::~TopK() {
}

void TopK::get_top_k(float* query, int k, int* indices) {
  int count = 0;
  const float* candidate;
  while (count < 1000000 && (candidate = prefetch_data_->get_data()) != NULL) {
    float score = 0;
    for (int i = 0; i < dimension_; ++i) {
      score += (query[i] - candidate[i]) * (query[i] - candidate[i]);
    }
    if (count < k) {
      item_[count].index = count;
      item_[count].score = score;
      if (count == k - 1)
        create_heap(k);
    } else {
      if (score < item_[0].score) {
        item_[0].index = count;
        item_[0].score = score;
        sort_heap(k, 0);
      }
    }
    ++count;
  }
  for (int i = 0; i < k; ++i) {
    indices[i] = item_[i].index;
  }
}

void TopK::create_heap(int k)
{
  int pos = (k - 1) / 2;
  for (int i = pos; i >= 0; --i)
    sort_heap(k, i);
}

void TopK::sort_heap(int k, int i)
{
  int t1, t2, pos;
  ITEM tmp;
  t1 = 2 * i + 1;
  t2 = t1 + 1;
  if (t1 > k - 1) {
    return ;
  } else {
    if (t2 > k - 1) {
      pos = t1;
    } else {
      pos = item_[t1].score > item_[t2].score? t1 : t2;
    }
    if (item_[i].score < item_[pos].score) {
      tmp.index = item_[i].index;
      tmp.score = item_[i].score;
      item_[i].index = item_[pos].index;
      item_[i].score = item_[pos].score;
      item_[pos].index = tmp.index;
      item_[pos].score = tmp.score;
      sort_heap(k, pos);
    }
  }
}
