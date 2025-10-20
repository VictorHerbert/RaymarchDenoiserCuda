#ifndef FILTER_H
#define FILTER_H

#include "utils.cuh"
#include "vector.cuh"
#include "image.cuh"

#include "third_party/helper_math.h"

struct DenoiseParams {
    union {
        int depth;
        int step;
    };
    float sigmaSpace;
    float sigmaColor;
    float sigmaAlbedo;
    float sigmaNormal;
};

KFUNC float gaussian(float p, float sigma);
KFUNC float gaussian(float2 p, float sigma);
KFUNC float gaussian(float3 p, float sigma);

KFUNC float lum(float3 col);

float3 snrCPU(Pixel* original, Pixel* noisy, int2 shape);
float3 snrGPU(Pixel* original, Pixel* noisy, int2 shape);

void waveletfilterCPU(Framebuffer frame, DenoiseParams params);
void waveletfilterGPU(Framebuffer frame, DenoiseParams params);
KERNEL void waveletKernel(Pixel* in, Pixel* out, Framebuffer frame, DenoiseParams params);
KFUNC  void waveletfilterPixel(int2 pos, Pixel* in, Pixel* out, Framebuffer frame, DenoiseParams params);


#endif