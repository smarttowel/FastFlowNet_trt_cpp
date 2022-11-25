#ifndef GRID_SAMPLER_CUH
#define GRID_SAMPLER_CUH

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "correlation.h"

static __forceinline__ __device__
    bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}


#endif