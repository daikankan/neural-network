#ifndef _PREFETCHINE_DATA_
#define _PREFETCHINE_DATA_

#include <vector>
#include <boost/smart_ptr.hpp>
#include "internal_thread.hpp"
#include "blocking_queue.hpp"

using std::vector;
using boost::shared_array;

class PrefetchingData : public InternalThread {
 public:
  explicit PrefetchingData(int prefetch_size, int dimension);
  virtual ~PrefetchingData();

 public:
  const float* get_data();

 protected:
  virtual void InternalThreadEntry();
  virtual bool load_data(float* data) = 0;

  vector<shared_array<float> > prefetch_;
  BlockingQueue<float*> prefetch_free_;
  BlockingQueue<float*> prefetch_full_;
  float* prefetch_current_;
};

#endif
