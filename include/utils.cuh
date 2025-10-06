#ifndef UTILS_H
#define UTILS_h

#include <chrono>

#define KERNEL __global__
#define KFUNC __forceinline__ __host__ __device__
#define LAUNCHER

#ifndef GL_RGB32F
#define GL_RGB32F 0x8815
#endif

typedef unsigned char uchar;

/*inline bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const int3& a, const int3& b) {
    return !(a == b);
}*/

//__device__ __host__ int dist(int2 p);
//__device__ __host__ float dist(float2 p);
//__device__ __host__ float dist(float3 p);
__device__ __host__ int totalSize(int2 shape);
__device__ __host__ int totalSize(int3 shape);
__device__ __host__ int inRange(int2 pos, int2 shape);
__device__ __host__ int index(int x, int y, int2 size);
__device__ __host__ int index(int2 p, int2 size);

#include <chrono>
#include <iostream>

#define MEASURE_TIME_MS(code_block, elapsed_var)           \
    do {                                                   \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block;                                        \
        auto end = std::chrono::high_resolution_clock::now();   \
        elapsed_var = std::chrono::duration<double, std::milli>(end - start).count(); \
    } while(0)


#endif