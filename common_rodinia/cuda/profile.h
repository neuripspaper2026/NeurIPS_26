#pragma once

#include "helper_cuda.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// NVTX is optional on some systems (e.g., GH200/GB200 images may not ship nvToolsExt).
#if __has_include(<nvToolsExt.h>)
  #include <nvToolsExt.h>
  #define HAS_NVTX 1
#elif __has_include("nvToolsExt.h")
  #include "nvToolsExt.h"
  #define HAS_NVTX 1
#else
  #define HAS_NVTX 0
  // Minimal no-op stubs so code compiles without NVTX.
  typedef int nvtxRangeId_t;
  static inline nvtxRangeId_t nvtxRangeStartA(const char*) { return 0; }
  static inline void nvtxRangeEnd(nvtxRangeId_t) {}
  static inline void nvtxMarkA(const char*) {}
#endif


extern bool enabled;
extern bool started;

#define PROFILE(launch)                                                        \
    {                                                                          \
        if (enabled && !started) {                                             \
            started = true;                                                    \
            checkCudaErrors(cudaProfilerStart());                              \
        }                                                                      \
        launch;                                                                \
    }
