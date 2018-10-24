#include <boost/thread.hpp>
#include <exception>
#include <iostream>

#include "internal_thread.hpp"

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  if (is_started()) {
    std::cout << "Threads should persist and not be restarted." << std::endl;
    return ;
  }
  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this));
  } catch (std::exception& e) {
    std::cout << "Thread exception: " << e.what() << std::endl;
  }
}

void InternalThread::entry() {
  InternalThreadEntry();
}

void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      std::cout << "Thread exception: " << e.what() << std::endl;
    }
  }
}
