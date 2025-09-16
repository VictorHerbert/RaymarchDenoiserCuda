#ifndef FILTER_H
#define FILTER_H

#include "matrix.cuh"

const int GAUSSIAN  = 0;
const int CROSS     = 1<<0;
const int BILATERAL = 1<<1;
const int WAVELET   = 1<<2;

__host__ __device__ float gaussian(float p, float sigma);
__host__ __device__ float gaussian(float2 p, float sigma);
__host__ __device__ float gaussian(float3 p, float sigma);


__host__ __device__ float lum(float3 col);

float3 snrCPU(float3* original, float3* noisy, int2 shape);
float3 snrGPU(float3* original, float3* noisy, int2 shape);

void waveletfilterCPU(
    int2 shape, int depth,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
);

void waveletfilterGPU(
    int2 shape, int depth,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
);

__global__ void waveletKernel(
    int2 shape, int step,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
);

__host__ __device__ void waveletfilterPixel(
    int2 pos, int2 shape, int step,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
);

#endif
