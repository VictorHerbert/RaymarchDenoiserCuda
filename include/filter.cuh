#ifndef FILTER_H
#define FILTER_H

#include "utils.cuh"
#include "vector.cuh"
#include "image.cuh"

#include "third_party/helper_math.h"

struct FilterParams {
    enum FilterType {AVERAGE, GAUSSIAN, CROSS, WAVELET} type;
    int depth;
    int step;
    float sigmaSpace;
    float sigmaColor;
    float sigmaAlbedo;
    float sigmaNormal;
};

CUDA_FUNC float lum(float3 col);

float3 snrCPU(Pixel* original, Pixel* noisy, int2 shape);
float3 snrGPU(Pixel* original, Pixel* noisy, int2 shape);

//void waveletfilterCPU(Framebuffer frame, FilterParams params);
//void waveletfilterGPU(Framebuffer frame, FilterParams params);

KERNEL void filterKernel(Framebuffer frame, FilterParams params);
CUDA_FUNC void filterPixel(int2 pos, const Pixel* in, Pixel* out, const Framebuffer frame, const FilterParams params);
CUDA_FUNC float waveletWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params);
CUDA_FUNC float averageWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params);

void waveletFilterSequence(std::string inputPath, std::string outputPath);

#endif