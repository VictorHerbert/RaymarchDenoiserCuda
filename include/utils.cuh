#ifndef UTILS_H
#define UTILS_h

#include <chrono>
#include <iostream>

#define KERNEL __global__
#define KFUNC __host__ __device__
#define LAUNCHER

#ifndef GL_RGB32F
#define GL_RGB32F 0x8815
#endif

typedef unsigned char uchar;

KFUNC int totalSize(int2 shape);

KFUNC int totalSize(int3 shape);

KFUNC int inRange(int2 pos, int2 shape);

KFUNC int index(int x, int y, int2 size);

KFUNC int index(int2 p, int2 size);

KFUNC uchar3 operator-(const uchar3 &a, const uchar3 &b);

KFUNC float length(const uchar3 &v);

KFUNC float dot(const uchar3 &a, const uchar3 &b);

KFUNC float3 operator*(const float &f, const uchar3 &v);

KFUNC float3 operator*(const uchar3 &v, const float &f);

KFUNC uchar3 make_uchar3(const float3 &v);

#define MEASURE_TIME_MS(code_block, elapsed_var)           \
    do {                                                   \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block;                                        \
        auto end = std::chrono::high_resolution_clock::now();   \
        elapsed_var = std::chrono::duration<double, std::milli>(end - start).count(); \
    } while(0)


//#include "third_party/helper_math.h"



#endif