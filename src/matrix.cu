#include "third_party/helper_math.h"

__device__ __host__ int totalSize(int2 shape){
    return shape.x * shape.y;
}

__device__ __host__ int totalSize(int3 shape){
    return shape.x * shape.y * shape.z;
}

__device__ __host__ int inRange(int2 pos, int2 shape){
    return (pos.x >= 0) && (pos.x < shape.x) && (pos.y >= 0) && (pos.y < shape.y);
}

__device__ __host__ int index(int2 p, int2 shape){
    return p.y * shape.x + p.x;
}

/*__device__ __host__ int dist(int2 p){
    return p.x*p.x + p.y*p.y;
}

__device__ __host__ float dist(float2 p){
    return p.x*p.x + p.y*p.y;
}

__device__ __host__ float dist(float3 p){
    return p.x*p.x + p.y*p.y + p.z*p.z;
}
*/