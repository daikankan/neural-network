#include <iostream>
#include "annoy.h"
#include "annoylib.h"
#include "kissrandom.h"

using std::vector;
using std::sort;
using std::less;

template<typename T>
class CompareIndicesByAnotherVectorValues {
  vector<T>* _values;
 public:
  CompareIndicesByAnotherVectorValues(vector<T>* values) : _values(values) {}
 public:
  bool operator() (const int& a, const int& b) const {
    return (_values)[a] < (_values)[b];
  }
};

int annoy_init(HANDLE* handle, int dimension) {

  *handle = static_cast<void*>(new AnnoyIndex<int, float, Angular,
                               Kiss32Random>(dimension));
  return 0;
}

int annoy_add_item(HANDLE handle, int index, float* data) {
  AnnoyIndex<int, float, Angular, Kiss32Random>* t = static_cast<
    AnnoyIndex<int, float, Angular, Kiss32Random>* >(handle);
  t->add_item(index, data);
  return 0;
}

int annoy_build_tree(HANDLE handle, int num_tree, bool save_tree) {
  AnnoyIndex<int, float, Angular, Kiss32Random>* t = static_cast<
    AnnoyIndex<int, float, Angular, Kiss32Random>* >(handle);
  t->build(num_tree);
  if (save_tree)
    t->save("megaface.tree");
  return 0;
}

int annoy_get_nns(HANDLE handle, int k, int n, float* query, int* res_indexs) {
  AnnoyIndex<int, float, Angular, Kiss32Random>* t = static_cast<
    AnnoyIndex<int, float, Angular, Kiss32Random>* >(handle);
  vector<int> toplist;
  vector<float> distance;
  t->get_nns_by_vector(query, n, (size_t)-1, &toplist, &distance);
  sort(toplist.begin(), toplist.end(),
       CompareIndicesByAnotherVectorValues<float>(&distance));
  for(int i = 0; i < k; ++i)
    res_indexs[i] = toplist[i];
  return 0;
}
