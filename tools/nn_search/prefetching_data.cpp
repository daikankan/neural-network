#include <boost/thread.hpp>
#include <iostream>
#include "prefetching_data.hpp"

PrefetchingData::PrefetchingData(int prefetch_size, int dimension)
    : prefetch_(prefetch_size), prefetch_free_(), prefetch_full_()
    , prefetch_current_(NULL) {
  for (int i = 0; i < prefetch_size; ++i) {
    prefetch_[i].reset(new float[dimension]);
    prefetch_free_.push(prefetch_[i].get());
  }
  StartInternalThread();
}

PrefetchingData::~PrefetchingData() {
}

const float* PrefetchingData:: get_data() {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("waiting for data...");
  return (const float*)prefetch_current_;
}

void PrefetchingData::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      float* data = prefetch_free_.pop("waiting for customer...");
      if (!load_data(data))
        break;
      prefetch_full_.push(data);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}
