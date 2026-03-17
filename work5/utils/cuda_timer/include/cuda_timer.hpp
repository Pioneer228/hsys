#ifndef _CUDA_TIMER_HPP_
#define _CUDA_TIMER_HPP_

#include <cuda_runtime.h>

class CUDATimer {
 private:
  float& _elapse_time_s;
  cudaEvent_t _start, _stop;

 public:
  CUDATimer(float& elapse_time_s);
  ~CUDATimer();
};

#endif  // _CUDA_TIMER_HPP_
