#ifndef UTILS_H
#define UTILS_h

#include <chrono>
#include <iostream>

#define KERNEL __global__
#define CUDA_FUNC __device__
#define CUDA_CPU_FUNC __device__ __host__
#define LAUNCHER

#ifndef GL_RGB32F
#define GL_RGB32F 0x8815
#endif

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA error: " << errStr << " at " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define MEASURE_TIME_MS(code_block, elapsed_var)           \
    do {                                                   \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block;                                        \
        auto end = std::chrono::high_resolution_clock::now();   \
        elapsed_var = std::chrono::duration<double, std::milli>(end - start).count(); \
    } while(0)

typedef unsigned char uchar;

CUDA_CPU_FUNC int totalSize(int2 shape);

CUDA_CPU_FUNC int totalSize(int3 shape);

CUDA_CPU_FUNC int inRange(int2 pos, int2 shape);

CUDA_CPU_FUNC int flattenIndex(int2 p, int2 size);

CUDA_CPU_FUNC uchar3 operator-(const uchar3 &a, const uchar3 &b);

CUDA_CPU_FUNC float length(const uchar3 &v);

CUDA_CPU_FUNC float dot(const uchar3 &a, const uchar3 &b);

CUDA_CPU_FUNC float3 operator*(const float &f, const uchar3 &v);

CUDA_CPU_FUNC float3 operator*(const uchar3 &v, const float &f);

CUDA_CPU_FUNC uchar3 make_uchar3(const float3 &v);

CUDA_CPU_FUNC float3 make_float3(const uchar3 &v);

CUDA_CPU_FUNC float length2(const float3 &v);

#endif