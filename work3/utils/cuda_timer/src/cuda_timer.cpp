#include "cuda_timer.hpp"

CUDATimer::CUDATimer(float& elapse_time_s)
    : _elapse_time_s(elapse_time_s) {
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  cudaEventRecord(_start);
}

CUDATimer::~CUDATimer() {
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, _start, _stop);

  _elapse_time_s = milliseconds / 1000;
}
