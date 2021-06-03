#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <hip/hip_runtime.h>

struct GpuTimer
{
  hipEvent_t start;
  hipEvent_t stop;

  GpuTimer()
  {
    hipEventCreate(&start);
    hipEventCreate(&stop);
  }

  ~GpuTimer()
  {
    hipEventDestroy(start);
    hipEventDestroy(stop);
  }

  void Start()
  {
    hipEventRecord(start, 0);
  }

  void Stop()
  {
    hipEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
