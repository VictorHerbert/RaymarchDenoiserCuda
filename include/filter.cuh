#ifndef FILTER_H
#define FILTER_H

#include "utils.cuh"
#include "vector.cuh"
#include "raymarch.cuh"

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

float3 snrCPU(float3* original, float3* noisy, int2 shape);
float3 snrGPU(float3* original, float3* noisy, int2 shape);

LAUNCHER    void waveletfilterCPU(Framebuffer frame, DenoiseParams params);
LAUNCHER    void waveletfilterGPU(Framebuffer frame, DenoiseParams params);

KERNEL      void waveletKernel(float3* in, float3* out, Framebuffer frame, DenoiseParams params);
KFUNC       void waveletfilterPixel(int2 pos, float3* in, float3* out, Framebuffer frame, DenoiseParams params);

#endif
