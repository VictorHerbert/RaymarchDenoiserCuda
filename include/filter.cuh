#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

struct FilterParams {
    enum FilterType {AVERAGE, GAUSSIAN, CROSS, WAVELET} type;
    int depth;
    int level;
    int radius;
    float sigmaSpace;
    float sigmaColor;
    float sigmaAlbedo;
    float sigmaNormal;

    bool cacheInput = true;
    bool cacheBuffer = true;
};

KERNEL void filterKernelBaseline(GBuffer frame, const FilterParams params);
KERNEL void filterKernelTiled(GBuffer frame, const FilterParams params);

CUDA_FUNC void cacheTile(uchar4* tile, uchar4* in, int2 shape, int radius);

#endif