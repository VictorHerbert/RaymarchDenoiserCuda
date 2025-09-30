#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>
#include <stdio.h>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;

inline bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const int3& a, const int3& b) {
    return !(a == b);
}

template <typename T>
struct CudaVector{
    T* data;
    int size;

    CudaVector(size_t size) : size(size) {
        cudaMalloc(&data, size * sizeof(T));
    }

    CudaVector(T* v, size_t size) : size(size) {
        cudaMalloc(&data, size * sizeof(T));
        //cudaMemcpy(data, v, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    CudaVector(std::vector<T>& v) : CudaVector(v.data(), v.size()){}

    void copy(std::vector<T>& v){
        //cudaMemcpy(data, v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~CudaVector() {
        cudaFree(data);
    }

};

__device__ __host__ int dist(int2 p);
//__device__ __host__ float dist(float2 p);
//__device__ __host__ float dist(float3 p);
__device__ __host__ int totalSize(int2 shape);
__device__ __host__ int totalSize(int3 shape);
__device__ __host__ int inRange(int2 pos, int2 shape);
__device__ __host__ int index(int x, int y, int2 size);
__device__ __host__ int index(int2 p, int2 size);


//__device__ __host__ float3 normalize(float3 in);
#endif